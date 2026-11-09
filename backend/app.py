# backend/app.py
import os
import sys
import time
import threading
import pickle
import shutil
from pathlib import Path
from glob import glob

import torch
import numpy as np
import librosa
import soundfile
from flask import Flask, request, jsonify, send_file, render_template

try:
    from flask_cors import CORS
except ImportError:
    print("[错误] 请安装 Flask-CORS: pip install Flask-CORS")
    sys.exit(1)

from CN_CustomTone_GlowTTS import (
    TTSConfig, ChineseProcessor, ChineseTokenizer, Model,
    get_loader, TTSDataset, list_model_packages as original_list_packages,
    synthesize_speech_from_package, load_model_package, generate_training_analysis
)

app = Flask(__name__, template_folder='../frontend')
CORS(app)

config = TTSConfig()

# 全局训练状态及线程锁
training_status = {
    "running": False,
    "progress": 0,
    "total_epochs": 0,
    "log": [],
    "model_name": None,
    "train_losses": [],
    "val_losses": [],
    "val_accuracies": [],
    "result_path": None,
    "start_time": None,
    "current_epoch": 0,
    "current_batch": 0,
    "total_batches": 0,
    "current_loss": 0.0,
    "batch_progress": 0,
    "analysis_images": []
}
status_lock = threading.Lock()

# 全局停止事件
stop_training_event = threading.Event()

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MELSPEC_DIR, exist_ok=True)
os.makedirs(config.MODEL_PACKAGE_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)


# ----------------------------- 辅助函数（训练回调） -----------------------------
def update_epoch_status(epoch, total_epochs, train_loss, val_loss, grad_norm, val_accuracy):
    with status_lock:
        training_status["progress"] = epoch
        training_status["total_epochs"] = total_epochs
        training_status["train_losses"].append(train_loss)
        training_status["val_losses"].append(val_loss)
        training_status["val_accuracies"].append(val_accuracy)
        log_entry = f"Epoch {epoch}/{total_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Acc: {val_accuracy:.4f}"
        if grad_norm is not None:
            log_entry += f" | Grad Norm: {grad_norm:.4f}"
        training_status["log"].append(log_entry)
        if len(training_status["log"]) > 200:
            training_status["log"] = training_status["log"][-200:]


def update_batch_status(epoch, total_epochs, batch, total_batches, loss):
    with status_lock:
        training_status["current_epoch"] = epoch
        training_status["total_epochs"] = total_epochs
        training_status["current_batch"] = batch
        training_status["total_batches"] = total_batches
        training_status["current_loss"] = loss
        training_status["batch_progress"] = int((batch / total_batches) * 100) if total_batches else 0
        if batch % 20 == 0 or batch == total_batches:
            training_status["log"].append(
                f"Epoch {epoch}/{total_epochs} | Batch {batch}/{total_batches} | Loss: {loss:.6f}")
            if len(training_status["log"]) > 200:
                training_status["log"] = training_status["log"][-200:]


def train_model_with_callbacks(config, model_name, stop_event, epoch_callback=None, batch_callback=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[训练] 使用设备: {device}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("[训练] 准备数据集...")
    train_dataset = TTSDataset(config, mode='train')

    # 强制从词汇表文件读取实际大小，解决缓存导致的不一致问题
    vocab_path = os.path.join(config.VOCAB_DIR, 'pinyin_vocab.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            config.PHONEME_VOCAB_SIZE = len(vocab)
            print(f"[训练] 从词汇表文件更新词汇表大小为: {config.PHONEME_VOCAB_SIZE}")

    actual_vocab_size = config.PHONEME_VOCAB_SIZE
    print(f"[训练] 词汇表大小: {actual_vocab_size}")

    model = Model(config)
    model.train()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[训练] 模型总参数量: {total_params / 1e6:.2f}M")

    try:
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=[config.BETA1, config.BETA2],
            weight_decay=config.WEIGHT_DECAY
        )
    except AttributeError:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=[config.BETA1, config.BETA2],
            weight_decay=config.WEIGHT_DECAY
        )

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    train_loader = get_loader(config, mode='train')
    val_loader = get_loader(config, mode='val')
    print(f"[训练] 训练集: {len(train_loader)} 批次, 验证集: {len(val_loader)} 批次")

    best_val_loss = float('inf')
    early_stop_counter = 0
    model_save_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    best_model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pth")

    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    grad_norm_history = []
    epoch_time_history = []

    initial_val_loss = None

    for epoch in range(config.NUM_EPOCHS):
        if stop_event.is_set():
            print("[训练] 收到停止信号，正在终止训练...")
            with status_lock:
                training_status["log"].append("⚠️ 用户手动停止训练")
            break

        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm_sum = 0.0
        num_batches = 0
        total_batches = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            if stop_event.is_set():
                print("[训练] 停止信号，批次循环中断")
                break

            for k in batch_data:
                batch_data[k] = batch_data[k].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                loss = model(**batch_data)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # 梯度裁剪及异常检测
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            grad_norm_val = grad_norm.item()
            if grad_norm_val > 100.0:  # 梯度爆炸检测
                print(f"[警告] 梯度范数异常 ({grad_norm_val:.2f})，跳过本次更新")
                optimizer.zero_grad()
                scaler.update()  # 关键：同步 scaler 状态，避免 unscale_ 重复调用错误
                continue

            epoch_grad_norm_sum += grad_norm_val
            num_batches += 1
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if batch_callback:
                batch_callback(epoch + 1, config.NUM_EPOCHS, batch_idx + 1, total_batches, loss.item())

        if stop_event.is_set():
            break

        train_avg_loss = epoch_loss / len(train_loader)
        avg_grad_norm = epoch_grad_norm_sum / num_batches if num_batches > 0 else 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                for k in batch_data:
                    batch_data[k] = batch_data[k].to(device, non_blocking=True)
                loss = model(**batch_data)
                val_loss += loss.item()
        val_avg_loss = val_loss / len(val_loader)

        if initial_val_loss is None:
            initial_val_loss = val_avg_loss
            val_accuracy = 0.0
        else:
            val_accuracy = max(0.0, 1.0 - val_avg_loss / initial_val_loss)

        # 早停判断
        if val_avg_loss < best_val_loss - config.EARLY_STOP_MIN_DELTA:
            best_val_loss = val_avg_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        train_loss_history.append(train_avg_loss)
        val_loss_history.append(val_avg_loss)
        val_accuracy_history.append(val_accuracy)
        grad_norm_history.append(avg_grad_norm)
        epoch_time_history.append(time.time() - epoch_start)

        if epoch_callback:
            epoch_callback(epoch + 1, config.NUM_EPOCHS, train_avg_loss, val_avg_loss, avg_grad_norm, val_accuracy)

        print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] Train Loss: {train_avg_loss:.6f} Val Loss: {val_avg_loss:.6f} Acc: {val_accuracy:.4f} Grad Norm: {avg_grad_norm:.4f}")

        # 早停触发
        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"[早停] 验证损失已连续 {config.EARLY_STOP_PATIENCE} 轮未改善，终止训练。")
            with status_lock:
                training_status["log"].append(f"⏹️ 早停触发，训练终止于第 {epoch+1} 轮")
            break

        scheduler.step()

    if stop_event.is_set():
        with status_lock:
            training_status["log"].append("训练已手动停止，当前模型已保存。")
    else:
        with status_lock:
            training_status["log"].append("训练正常完成！")

    # 保存最终模型
    torch.save(model.state_dict(), model_save_path)

    # 打包模型（自包含）
    package_dir = config.get_model_package_path(model_name)
    shutil.copy2(best_model_path, os.path.join(package_dir, f"{model_name}_best.pth"))
    shutil.copy2(model_save_path, os.path.join(package_dir, f"{model_name}.pth"))
    vocab_source = os.path.join(config.VOCAB_DIR, 'pinyin_vocab.pkl')
    if os.path.exists(vocab_source):
        shutil.copy2(vocab_source, os.path.join(package_dir, 'vocab.pkl'))
    else:
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_source}")

    # 从实际保存的 vocab.pkl 重新读取词汇表大小
    with open(os.path.join(package_dir, 'vocab.pkl'), 'rb') as f:
        saved_vocab = pickle.load(f)
        real_vocab_size = len(saved_vocab)

    config_to_save = {
        'SAMPLE_RATE': config.SAMPLE_RATE,
        'N_FFT': config.N_FFT,
        'HOP_LENGTH': config.HOP_LENGTH,
        'N_MELS': config.N_MELS,
        'FMIN': config.FMIN,
        'EMBEDDING_DIM': config.EMBEDDING_DIM,
        'ENCODER_CONV_FILTERS': config.ENCODER_CONV_FILTERS,
        'ENCODER_NUM_LAYERS': config.ENCODER_NUM_LAYERS,
        'DECODER_NUM_FLOWS': config.DECODER_NUM_FLOWS,
        'PHONEME_VOCAB_SIZE': real_vocab_size,
        'DATASET_PATH': config.DATASET_PATH,
        'Model_Name': model_name,
        'Training_Time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Final_Train_Loss': train_loss_history[-1] if train_loss_history else None,
        'Final_Val_Loss': val_loss_history[-1] if val_loss_history else None,
    }
    with open(os.path.join(package_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config_to_save, f)

    plot_dir = os.path.join(config.OUTPUT_DIR, 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    if train_loss_history:
        generate_training_analysis(
            train_loss_history, val_loss_history, val_accuracy_history,
            grad_norm_history, epoch_time_history,
            config, model_name, plot_dir, best_val_loss
        )
    analysis_images = []
    for f in os.listdir(plot_dir):
        if f.endswith('.png') and model_name in f:
            analysis_images.append(f"/outputs/training_plots/{f}")
    report_file = f"convergence_report_{model_name}.txt"
    if os.path.exists(os.path.join(plot_dir, report_file)):
        analysis_images.append(f"/outputs/training_plots/{report_file}")

    return package_dir, analysis_images


def training_worker(model_name, dataset_path, epochs, batch_size, lr):
    global training_status, config, stop_training_event
    try:
        if dataset_path and os.path.exists(dataset_path):
            config.DATASET_PATH = dataset_path
        config.NUM_EPOCHS = epochs
        config.BATCH_SIZE = batch_size
        config.LEARNING_RATE = lr

        stop_training_event.clear()

        with status_lock:
            training_status.update({
                "running": True,
                "progress": 0,
                "total_epochs": epochs,
                "log": [],
                "model_name": model_name,
                "train_losses": [],
                "val_losses": [],
                "val_accuracies": [],
                "result_path": None,
                "start_time": time.time(),
                "current_epoch": 0,
                "current_batch": 0,
                "total_batches": 0,
                "current_loss": 0.0,
                "batch_progress": 0,
                "analysis_images": []
            })

        def epoch_cb(epoch, total, train_loss, val_loss, grad_norm, val_accuracy):
            update_epoch_status(epoch, total, train_loss, val_loss, grad_norm, val_accuracy)

        def batch_cb(epoch, total, batch, total_batches, loss):
            update_batch_status(epoch, total, batch, total_batches, loss)

        package_dir, analysis_images = train_model_with_callbacks(
            config, model_name, stop_training_event,
            epoch_callback=epoch_cb, batch_callback=batch_cb
        )
        with status_lock:
            training_status["result_path"] = package_dir
            training_status["analysis_images"] = analysis_images
            training_status["running"] = False
            training_status["log"].append(f"训练完成！模型已打包至 {package_dir}")
    except Exception as e:
        with status_lock:
            training_status["running"] = False
            training_status["log"].append(f"训练失败: {str(e)}")
        print(f"训练线程异常: {e}")
    finally:
        stop_training_event.clear()


# ----------------------------- API 路由 -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/list_models', methods=['GET'])
def list_models():
    packages, _ = original_list_packages(config)
    return jsonify({"models": packages})


@app.route('/api/train', methods=['POST'])
def start_training():
    global training_status
    with status_lock:
        if training_status["running"]:
            return jsonify({"error": "已有训练任务正在运行"}), 409

    data = request.json
    model_name = data.get('model_name', 'my_tts_model')
    dataset_path = data.get('dataset_path', config.DATASET_PATH)
    epochs = int(data.get('epochs', config.NUM_EPOCHS))
    batch_size = int(data.get('batch_size', config.BATCH_SIZE))
    lr = float(data.get('lr', config.LEARNING_RATE))

    thread = threading.Thread(
        target=training_worker,
        args=(model_name, dataset_path, epochs, batch_size, lr),
        daemon=True
    )
    thread.start()

    return jsonify({"message": "训练已启动", "status_url": "/api/train_status"})


@app.route('/api/stop_train', methods=['POST'])
def stop_training():
    global stop_training_event, training_status
    with status_lock:
        if not training_status["running"]:
            return jsonify({"error": "当前没有正在运行的训练任务"}), 400
    stop_training_event.set()
    with status_lock:
        training_status["log"].append("⏸️ 正在请求停止训练...")
    return jsonify({"message": "已发送停止信号，训练将在当前批次结束后停止"})


@app.route('/api/train_status', methods=['GET'])
def get_training_status():
    with status_lock:
        return jsonify({
            "running": training_status["running"],
            "progress": training_status["progress"],
            "total_epochs": training_status["total_epochs"],
            "current_epoch": training_status.get("current_epoch", 0),
            "current_batch": training_status.get("current_batch", 0),
            "total_batches": training_status.get("total_batches", 0),
            "current_loss": training_status.get("current_loss", 0.0),
            "batch_progress": training_status.get("batch_progress", 0),
            "log": training_status["log"][-100:],
            "train_losses": training_status["train_losses"],
            "val_losses": training_status["val_losses"],
            "val_accuracies": training_status.get("val_accuracies", []),
            "analysis_images": training_status.get("analysis_images", [])
        })


@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    model_name = data.get('model_name')
    text = data.get('text', '')
    volume = float(data.get('volume', 1.0))
    pitch = int(data.get('pitch', 0))
    speed = float(data.get('speed', 1.0))

    if not model_name:
        return jsonify({"error": "请选择模型包"}), 400

    package_path = os.path.join(config.MODEL_PACKAGE_DIR, model_name)
    if not os.path.exists(package_path):
        return jsonify({"error": f"模型包不存在: {model_name}"}), 404

    try:
        output_wav, mel_path = synthesize_speech_from_package(
            package_path, text, save_mel=True,
            volume=volume, pitch_shift=pitch, speed_rate=speed
        )
        model_name_enc = model_name
        timestamp_dir = os.path.basename(os.path.dirname(output_wav))
        wav_filename = os.path.basename(output_wav)
        audio_url = f"/outputs/{model_name_enc}/{timestamp_dir}/{wav_filename}"
        mel_url = None
        if mel_path:
            mel_filename = os.path.basename(mel_path)
            mel_url = f"/outputs/{model_name_enc}/{timestamp_dir}/{mel_filename}"

        return jsonify({
            "audio_url": audio_url,
            "mel_url": mel_url,
            "local_path": output_wav
        })
    except Exception as e:
        return jsonify({"error": f"合成异常: {str(e)}"}), 500


# ----------------------------- 客观评测 API -----------------------------
@app.route('/api/upload_ref', methods=['POST'])
def upload_ref_audio():
    if 'file' not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    temp_dir = os.path.join(config.OUTPUT_DIR, 'temp_eval')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"ref_{int(time.time())}.wav")
    file.save(temp_path)
    return jsonify({"temp_path": temp_path})


@app.route('/api/evaluate', methods=['POST'])
def evaluate_speech():
    data = request.json
    syn_path = data.get('syn_audio_path')
    ref_path = data.get('ref_audio_path')

    if not syn_path or not os.path.exists(syn_path):
        return jsonify({"error": "合成音频路径无效"}), 400
    if not ref_path or not os.path.exists(ref_path):
        return jsonify({"error": "参考音频路径无效"}), 400

    try:
        wav_syn, sr_syn = soundfile.read(syn_path, dtype='float32')
        wav_ref, sr_ref = soundfile.read(ref_path, dtype='float32')

        if sr_syn != config.SAMPLE_RATE:
            wav_syn = librosa.resample(wav_syn, orig_sr=sr_syn, target_sr=config.SAMPLE_RATE)
        if sr_ref != config.SAMPLE_RATE:
            wav_ref = librosa.resample(wav_ref, orig_sr=sr_ref, target_sr=config.SAMPLE_RATE)

        processor = ChineseProcessor(config)

        if np.array_equal(wav_ref, wav_syn):
            return jsonify({"error": "参考音频和合成音频为同一文件，无法评测"}), 400

        mcd = processor.calculate_mcd(wav_ref, wav_syn)
        stoi_score = processor.calculate_stoi(wav_ref, wav_syn, sr=config.SAMPLE_RATE)

        if 'temp_eval' in ref_path:
            try:
                os.remove(ref_path)
            except:
                pass

        return jsonify({
            "mcd": float(mcd),
            "stoi": float(stoi_score) if stoi_score is not None else None,
            "message": "评测完成"
        })
    except Exception as e:
        return jsonify({"error": f"评测失败: {str(e)}"}), 500


# ----------------------------- 静态文件路由 -----------------------------
@app.route('/outputs/<model_name>/<timestamp>/<filename>')
def serve_output_file(model_name, timestamp, filename):
    safe_path = os.path.join(config.OUTPUT_DIR, model_name, timestamp, filename)
    if not os.path.exists(safe_path):
        return "文件不存在", 404
    mimetype = 'audio/wav' if filename.endswith('.wav') else 'image/png'
    return send_file(safe_path, mimetype=mimetype)


@app.route('/outputs/<filename>')
def serve_audio_legacy(filename):
    safe_path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(safe_path):
        return "文件不存在", 404
    return send_file(safe_path, mimetype='audio/wav')


@app.route('/melspec/<filename>')
def serve_melspec(filename):
    safe_path = os.path.join(config.MELSPEC_DIR, filename)
    if not os.path.exists(safe_path):
        return "文件不存在", 404
    return send_file(safe_path, mimetype='image/png')


@app.route('/outputs/training_plots/<path:filename>')
def serve_training_plot(filename):
    safe_path = os.path.join(config.OUTPUT_DIR, 'training_plots', filename)
    if not os.path.exists(safe_path):
        return "文件不存在", 404
    return send_file(safe_path)


# ----------------------------- 启动应用 -----------------------------
if __name__ == '__main__':
    if not os.path.exists(app.template_folder):
        os.makedirs(app.template_folder, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)