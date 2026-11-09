# ================================
# 文件: CN_CustomTone_GlowTTS.py
# 描述: 交互式中文TTS框架（稳定收敛版，回退至训练曲线完美版本）
# ================================

import numpy as np
import torch
import pickle
import os
import re
import librosa
import soundfile
import scipy.io.wavfile
from tqdm import tqdm
from pathlib import Path
from pypinyin import pinyin, Style
import matplotlib.pyplot as plt
import matplotlib
import math
from glob import glob
import time
import shutil

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
PYPIYIN_AVAILABLE = True


class TTSConfig:
    # 路径配置
    ROOT_DIR = './save/'
    CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
    CACHE_TEXT_PATH = os.path.join(CACHE_DIR, 'text')
    CACHE_MEL_PATH = os.path.join(CACHE_DIR, 'mel')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
    VOCAB_DIR = os.path.join(ROOT_DIR, 'vocab')
    MELSPEC_DIR = os.path.join(ROOT_DIR, 'melspec')
    MODEL_PACKAGE_DIR = os.path.join(ROOT_DIR, 'model_packages')

    # 音频处理参数
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 80
    FMIN = 50

    # 模型结构
    EMBEDDING_DIM = 192
    ENCODER_CONV_FILTERS = 192
    ENCODER_NUM_LAYERS = 4
    DECODER_NUM_FLOWS = 6
    PHONEME_VOCAB_SIZE = 179

    # 训练基础配置（已调整为稳定收敛参数）
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    BETA1, BETA2 = 0.9, 0.999
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 30
    EARLY_STOP_MIN_DELTA = 1e-5

    WARMUP_STEPS = 4
    LOSS_W_RECON = 1.0
    LOSS_W_DUR = 0.3

    INFERENCE_TEXT = "你好，语音合成测试。"
    VOLUME = 1.0
    PITCH_SHIFT = 0
    SPEED_RATE = 1.0
    DATASET_PATH = ROOT_DIR

    # 可切换相位重构方法（True = Griffin‑Lim，False = ADMM）
    USE_GRIFFIN_LIM = False

    def get_model_package_path(self, model_name):
        package_dir = os.path.join(self.MODEL_PACKAGE_DIR, model_name)
        os.makedirs(package_dir, exist_ok=True)
        return package_dir


class ChineseProcessor:
    def __init__(self, config):
        self.config = config
        self.mel_basis = librosa.filters.mel(
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=None
        )
        os.makedirs(config.MELSPEC_DIR, exist_ok=True)

    def load_text(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"[Error] 读取文本 {txt_path} 失败: {e}")
            return None

    def _find_audio_text_pairs(self, data_root):
        wav_files = list(Path(data_root).rglob("*.wav"))
        pairs = []
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix('.txt')
            if txt_path.exists():
                pairs.append((str(wav_path), str(txt_path)))
            else:
                print(f"[Warning] 未找到与 {wav_path} 对应的文本文件，已跳过")
        print(f"[Processor] 在 {data_root} 下找到 {len(pairs)} 个有效音频-文本对")
        return pairs

    def load_and_process_audio(self, wav_path):
        try:
            wav, sr = soundfile.read(wav_path, dtype='float32')
            if sr != self.config.SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        except Exception as e:
            print(f"[Error] 读取音频 {wav_path} 失败: {e}")
            return None
        wav = librosa.effects.trim(wav[220:-220], top_db=45, frame_length=1024, hop_length=256)[0]
        stft = librosa.stft(y=wav, n_fft=self.config.N_FFT, hop_length=self.config.HOP_LENGTH,
                            win_length=self.config.WIN_LENGTH, pad_mode='reflect', window='hann', center=True)
        mel = np.dot(self.mel_basis, np.abs(stft))
        mel = np.maximum(1e-8, mel)
        mel = np.log10(mel) * 20.0 - 20
        mel = (mel + 100) / 100
        mel = 8.0 * mel - 4.0
        mel = np.clip(mel, -4.0, 4.0)
        return mel.astype(np.float32)

    def _admm_phase_reconstruction(self, magnitude, n_iter=35, rho=0.15, gamma=0.8):
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        X = magnitude * angles
        V = X.copy()
        U = np.zeros_like(X)
        for _ in range(n_iter):
            Z = V - U
            X_new = (2 * magnitude * np.exp(1j * np.angle(Z)) + rho * Z) / (2 + rho)
            X = X_new
            wav = librosa.istft(V, hop_length=self.config.HOP_LENGTH, win_length=self.config.WIN_LENGTH,
                                center=True, window='hann')
            if gamma > 0:
                wav = librosa.effects.preemphasis(wav, coef=0.97)
                wav = librosa.effects.deemphasis(wav, coef=0.97)
            V_new = librosa.stft(wav, n_fft=self.config.N_FFT, hop_length=self.config.HOP_LENGTH,
                                 win_length=self.config.WIN_LENGTH, center=True, window='hann')
            V = V_new
            U = U + (X - V)
        wav = librosa.istft(V, hop_length=self.config.HOP_LENGTH, win_length=self.config.WIN_LENGTH,
                            center=True, window='hann')
        return wav

    def _griffin_lim_reconstruction(self, magnitude, n_iter=60):
        """Griffin‑Lim 相位重构"""
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        stft_complex = magnitude * angles
        for _ in range(n_iter):
            wav = librosa.istft(stft_complex, hop_length=self.config.HOP_LENGTH,
                                win_length=self.config.WIN_LENGTH, center=True, window='hann')
            stft_complex = librosa.stft(wav, n_fft=self.config.N_FFT,
                                        hop_length=self.config.HOP_LENGTH,
                                        win_length=self.config.WIN_LENGTH,
                                        center=True, window='hann')
            stft_complex = magnitude * np.exp(1j * np.angle(stft_complex))
        wav = librosa.istft(stft_complex, hop_length=self.config.HOP_LENGTH,
                            win_length=self.config.WIN_LENGTH, center=True, window='hann')
        return wav

    def mel_to_wav(self, mel, volume=None, pitch_shift=None, speed_rate=None):
        volume = volume if volume is not None else self.config.VOLUME
        pitch_shift = pitch_shift if pitch_shift is not None else self.config.PITCH_SHIFT
        speed_rate = speed_rate if speed_rate is not None else self.config.SPEED_RATE

        mel = np.clip(mel, -4.0, 4.0)
        mel = ((mel + 4.0) * 100 / 8.0) - 100
        mel = np.power(10, (mel + 20) / 20.0)
        inv_mel_basis = np.linalg.pinv(self.mel_basis)
        magnitude = np.maximum(1e-10, np.dot(inv_mel_basis, mel))

        if self.config.USE_GRIFFIN_LIM:
            print("[合成] 使用 Griffin‑Lim 重构相位")
            wav = self._griffin_lim_reconstruction(magnitude, n_iter=60)
        else:
            wav = self._admm_phase_reconstruction(magnitude, n_iter=35, rho=0.15, gamma=0.8)

        wav = wav * volume
        if pitch_shift != 0:
            wav = librosa.effects.pitch_shift(wav, sr=self.config.SAMPLE_RATE, n_steps=pitch_shift)
        if speed_rate != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=speed_rate)
        wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        return wav.astype(np.int16)

    def save_mel_spectrogram(self, mel, output_path, title="梅尔频谱图"):
        if len(mel.shape) == 3:
            mel = mel[0]
        if mel.shape[0] != self.config.N_MELS:
            mel = mel.T
        mel_viz = np.clip(mel, -4.0, 4.0)
        mel_viz = ((mel_viz + 4.0) * 100 / 8.0) - 100
        mel_viz = np.power(10, (mel_viz + 20) / 20.0)
        mel_db = librosa.power_to_db(mel_viz, ref=np.max)
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mel_db, sr=self.config.SAMPLE_RATE, hop_length=self.config.HOP_LENGTH,
                                 x_axis='time', y_axis='mel', fmin=self.config.FMIN)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def denoise_wav(self, wav):
        original_dtype = wav.dtype
        if original_dtype == np.int16:
            wav_float = wav.astype(np.float32) / 32768.0
        elif original_dtype == np.float32:
            wav_float = wav.copy()
        else:
            raise ValueError(f"不支持的音频数据类型: {original_dtype}")
        try:
            import noisereduce as nr
            wav_denoised = nr.reduce_noise(y=wav_float, sr=self.config.SAMPLE_RATE,
                                           prop_decrease=0.5, stationary=False)
        except ImportError:
            n_fft = 2048
            hop_length = 512
            win_length = 2048
            D = librosa.stft(wav_float, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            magnitude, phase = librosa.magphase(D)
            noise_frames = magnitude[:, :10]
            noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
            over_subtraction_factor = 1.5
            spectral_floor = 0.3
            magnitude_denoised = magnitude - over_subtraction_factor * noise_profile
            magnitude_floor = spectral_floor * magnitude
            magnitude_denoised = np.maximum(magnitude_denoised, magnitude_floor)
            D_denoised = magnitude_denoised * phase
            wav_denoised = librosa.istft(D_denoised, hop_length=hop_length, win_length=win_length)
        min_len = min(len(wav_float), len(wav_denoised))
        wav_denoised = wav_denoised[:min_len]
        if original_dtype == np.int16:
            wav_denoised = (wav_denoised * 32767.0).clip(-32768, 32767).astype(np.int16)
        else:
            wav_denoised = wav_denoised.astype(np.float32)
        return wav_denoised

    def _compute_mel(self, wav):
        stft = librosa.stft(wav, n_fft=self.config.N_FFT,
                            hop_length=self.config.HOP_LENGTH,
                            win_length=self.config.WIN_LENGTH,
                            center=True, window='hann')
        mag = np.abs(stft)
        mel = np.dot(self.mel_basis, mag)
        mel = np.maximum(mel, 1e-8)
        mel = np.log10(mel) * 20.0 - 20
        return mel.astype(np.float32)

    def calculate_mcd(self, wav_ref, wav_syn):
        if wav_ref.dtype == np.int16:
            wav_ref = wav_ref.astype(np.float32) / 32768.0
        if wav_syn.dtype == np.int16:
            wav_syn = wav_syn.astype(np.float32) / 32768.0
        min_len = min(len(wav_ref), len(wav_syn))
        wav_ref = wav_ref[:min_len]
        wav_syn = wav_syn[:min_len]
        mel_ref = self._compute_mel(wav_ref)
        mel_syn = self._compute_mel(wav_syn)
        min_frames = min(mel_ref.shape[1], mel_syn.shape[1])
        mel_ref = mel_ref[:, :min_frames]
        mel_syn = mel_syn[:, :min_frames]
        frame_dist = np.sqrt(np.sum((mel_ref - mel_syn) ** 2, axis=0))
        mcd = np.mean(frame_dist)
        return float(mcd)

    def calculate_stoi(self, wav_ref, wav_syn, sr=None):
        try:
            from pystoi import stoi
            if sr is None:
                sr = self.config.SAMPLE_RATE
            if wav_ref.dtype == np.int16:
                wav_ref = wav_ref.astype(np.float32) / 32768.0
            if wav_syn.dtype == np.int16:
                wav_syn = wav_syn.astype(np.float32) / 32768.0
            min_len = min(len(wav_ref), len(wav_syn))
            wav_ref = wav_ref[:min_len]
            wav_syn = wav_syn[:min_len]
            score = stoi(wav_ref, wav_syn, sr, extended=False)
            return float(score) if score is not None else None
        except ImportError:
            return None


class ChineseTokenizer:
    def __init__(self, config, vocab_filename='pinyin_vocab.pkl'):
        self.config = config
        self.vocab_save_path = os.path.join(config.VOCAB_DIR, vocab_filename)
        os.makedirs(config.VOCAB_DIR, exist_ok=True)
        self.vocab = {'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, '<UNK>': 4}
        punctuation_map = {
            ',': 5, '.': 6, '?': 7, '!': 8, ';': 9, ':': 10, '-': 11, '"': 12, "'": 13,
            '(': 14, ')': 15, '[': 16, ']': 17, '{': 18, '}': 19, '<': 20, '>': 21,
            '，': 22, '。': 23, '？': 24, '！': 25, '；': 26, '：': 27, '、': 28,
            '「': 29, '」': 30, '『': 31, '』': 32, '（': 33, '）': 34, '【': 35, '】': 36,
            '《': 37, '》': 38, '—': 39, '～': 40, '·': 41, '…': 42,
            ' ': 43, '\t': 44, '\n': 45
        }
        self.vocab.update(punctuation_map)
        for i in range(10):
            self.vocab[str(i)] = 46 + i
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.vocab[c] = 56 + i
        self.next_id = 106
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._load_vocab()

    def _load_vocab(self):
        if os.path.exists(self.vocab_save_path):
            print(f"[Tokenizer] 加载词汇表: {self.vocab_save_path}")
            with open(self.vocab_save_path, 'rb') as f:
                saved_vocab = pickle.load(f)
                self.vocab.update(saved_vocab)
                self.id_to_token = {v: k for k, v in self.vocab.items()}
                self.next_id = max(self.vocab.values()) + 1
            print(f"[Tokenizer] 词汇表大小: {len(self.vocab)}")
        else:
            print(f"[Tokenizer] 创建新词汇表")

    def save_vocab(self):
        with open(self.vocab_save_path, 'wb') as f:
            pickle.dump(self.vocab, f)

    def _chinese_to_pinyin(self, text):
        if not PYPIYIN_AVAILABLE:
            raise ImportError("请安装pypinyin: pip install pypinyin")
        text = re.sub(r'[<>\(\)\[\]"]+', '', text)
        pinyin_list = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors='default')
        result = []
        for item in pinyin_list:
            if item:
                py = item[0]
                if re.match(r'^[a-zA-Z0-9\s\.,!?;:\'"\-]+$', py):
                    if py.strip() == '':
                        result.append(' ')
                    else:
                        for char in py:
                            if char.strip():
                                result.append(char)
                else:
                    result.append(py)
        return result

    def _get_or_add_token_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
            return self.vocab[token]

    def __call__(self, text, update_vocab=False):
        if not text or not text.strip():
            return []
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        try:
            pinyin_seq = self._chinese_to_pinyin(text)
        except Exception as e:
            print(f"[Error] 拼音转换失败: {e}, 文本: '{text}'")
            return []
        ids = []
        for py in pinyin_seq:
            if update_vocab:
                token_id = self._get_or_add_token_id(py)
            else:
                token_id = self.vocab.get(py, self.vocab['<UNK>'])
            ids.append(token_id)
        return ids

    def decode(self, ids):
        tokens = []
        for id_val in ids:
            tokens.append(self.id_to_token.get(id_val, '<UNK_ID>'))
        return ' '.join(tokens)


class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.cache_text = []
        self.cache_mel = []
        self.processor = ChineseProcessor(config)
        self.tokenizer = ChineseTokenizer(config)
        os.makedirs(config.CACHE_TEXT_PATH, exist_ok=True)
        os.makedirs(config.CACHE_MEL_PATH, exist_ok=True)
        cache_text_file = os.path.join(config.CACHE_TEXT_PATH, f'text_cache_{mode}.pkl')
        cache_mel_file = os.path.join(config.CACHE_MEL_PATH, f'mel_cache_{mode}.pkl')
        if os.path.exists(cache_text_file) and os.path.exists(cache_mel_file):
            print(f"[Dataset-{mode}] 加载缓存数据...")
            with open(cache_text_file, 'rb') as f:
                self.cache_text = pickle.load(f)
            with open(cache_mel_file, 'rb') as f:
                self.cache_mel = pickle.load(f)
            return
        print(f"[Dataset-{mode}] 构建数据集...")
        pairs = self.processor._find_audio_text_pairs(config.DATASET_PATH)
        if len(pairs) == 0:
            raise FileNotFoundError(f"未找到任何音频-文本对！")
        if mode == 'train':
            pairs = pairs[:int(len(pairs) * 0.9)]
        elif mode == 'val':
            pairs = pairs[int(len(pairs) * 0.9):]
        valid_count = 0
        for wav_path, txt_path in tqdm(pairs, desc=f"处理{mode}数据"):
            mel = self.processor.load_and_process_audio(wav_path)
            if mel is None or mel.shape[1] == 0:
                continue
            text_raw = self.processor.load_text(txt_path)
            if not text_raw:
                continue
            update_vocab = (mode == 'train')
            token_ids = self.tokenizer(text_raw, update_vocab=update_vocab)
            if 0 < len(token_ids) <= 200:
                token_ids = [self.tokenizer.vocab['<BOS>']] + token_ids + [self.tokenizer.vocab['<EOS>']]
                self.cache_text.append(np.array(token_ids, dtype=np.int32))
                self.cache_mel.append(mel)
                valid_count += 1
        with open(cache_text_file, 'wb') as f:
            pickle.dump(self.cache_text, f)
        with open(cache_mel_file, 'wb') as f:
            pickle.dump(self.cache_mel, f)
        if mode == 'train':
            self.tokenizer.save_vocab()
            self.config.PHONEME_VOCAB_SIZE = len(self.tokenizer.vocab)

    def __len__(self):
        return len(self.cache_text)

    def __getitem__(self, idx):
        return {'text': self.cache_text[idx], 'mel': self.cache_mel[idx]}


def get_loader(config, mode='train'):
    def collate_fn(data):
        batch_size = len(data)
        text_list = [sample['text'] for sample in data]
        mel_list = [sample['mel'] for sample in data]
        len_text = [len(t) for t in text_list]
        len_mel = [m.shape[1] for m in mel_list]
        text_padded = np.zeros((batch_size, max(len_text)), dtype='int32')
        for i in range(batch_size):
            text_padded[i, :len_text[i]] = text_list[i]
        mel_padded = np.zeros((batch_size, config.N_MELS, max(len_mel)))
        for i in range(batch_size):
            mel_padded[i, :, :len_mel[i]] = mel_list[i]
        return {
            'text': torch.LongTensor(text_padded),
            'lens_text': torch.LongTensor(len_text),
            'mel': torch.FloatTensor(mel_padded),
            'lens_mel': torch.LongTensor(len_mel),
        }
    dataset = TTSDataset(config, mode=mode)
    shuffle = (mode == 'train')
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=(mode == 'train'),
        num_workers=0,
        pin_memory=True
    )


def get_mask(lens):
    mask = torch.zeros(len(lens), max(lens), dtype=torch.bool, device=lens.device)
    for i in range(len(lens)):
        mask[i, :lens[i]] = True
    return mask


def maximum_path(atten, mask):
    atten = atten * mask
    device = atten.device
    b, lens_text, lens_mel = atten.shape
    dtype = atten.dtype
    atten_np = atten.cpu().detach().numpy()
    mask_np = mask.cpu().detach().numpy().astype(np.int64)
    direction = np.zeros((b, lens_text, lens_mel), dtype=np.int64)
    value = np.zeros((b, lens_text))
    for i in range(lens_mel):
        value_shift = np.pad(value, [[0, 0], [1, 0]], mode='constant', constant_values=-np.inf)[:, :lens_text]
        direction[:, :, i] = value >= value_shift
        select = np.ones([1, lens_text], np.int64)
        select[:, i + 1:] = 0
        value = np.where(value >= value_shift, value, value_shift)
        value = np.where(select, value + atten_np[:, :, i], -np.inf)
    direction = np.where(mask_np, direction, 1)
    path = np.zeros([b, lens_text, lens_mel])
    select = mask_np[:, :, 0].sum(1) - 1
    for i in range(lens_mel - 1, -1, -1):
        path[np.arange(b), select, i] = 1
        select = select + direction[np.arange(b), select, i] - 1
    path = path * mask_np.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logs = torch.nn.Parameter(torch.zeros(1, 160, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 160, 1))

    def forward(self, x, mask, inference=False):
        lens = mask.sum([1, 2])
        if inference:
            x = (x - self.bias) * (-self.logs).exp() * mask
        else:
            x = (self.bias + self.logs.exp() * x) * mask
        logdet = self.logs.sum() * lens
        return x, logdet


class WN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.05)
        self.cnn1 = torch.nn.ModuleList([torch.nn.Conv1d(192, 384, 5, dilation=1, padding=2) for _ in range(4)])
        self.cnn2 = torch.nn.ModuleList(
            [torch.nn.Conv1d(192, 384, 1) for _ in range(3)] + [torch.nn.Conv1d(192, 192, 1)])
        for i in range(4):
            self.cnn1[i] = torch.nn.utils.parametrizations.weight_norm(self.cnn1[i], name='weight')
            self.cnn2[i] = torch.nn.utils.parametrizations.weight_norm(self.cnn2[i], name='weight')

    def forward(self, mel, mask):
        out = 0
        for i in range(4):
            x = self.cnn1[i](mel)
            x = self.dropout(x)
            x = x[:, :192].tanh() * x[:, 192:].sigmoid()
            x = self.cnn2[i](x)
            mel = (mel + x[:, :192]) * mask
            out = out + (x[:, 192:] if i != 3 else x)
        return out * mask


class CouplingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        start = torch.nn.Conv1d(80, 192, 1)
        start = torch.nn.utils.parametrizations.weight_norm(start)
        self.start = start
        end = torch.nn.Conv1d(192, 160, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.wn = WN()

    def forward(self, mel, mask, inference=False):
        res = mel
        mel = self.start(mel[:, :80]) * mask
        mel = self.wn(mel, mask)
        mel = self.end(mel)
        if inference:
            out = (res[:, 80:] - mel[:, :80]) * (-mel[:, 80:]).exp() * mask
        else:
            out = (mel[:, :80] + mel[:, 80:].exp() * res[:, 80:]) * mask
        logdet = (mel[:, 80:] * mask).sum([1, 2])
        out = torch.cat([res[:, :80], out], 1)
        return out, logdet


class InvConvNear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight, _ = torch.linalg.qr(torch.randn(4, 4), 'complete')
        weight[:, 0] = -weight[:, 0]
        self.weight = torch.nn.Parameter(weight)

    def forward(self, mel, mask, inference=False):
        b, _, dim = mel.shape
        mel = mel.reshape(b, 2, 40, 2, dim).permute(0, 1, 3, 2, 4).reshape(b, 4, 40, dim)
        weight = self.weight.inverse() if inference else self.weight
        logdet = self.weight.logdet() * 40 * mask.sum([1, 2])
        weight = weight.reshape(4, 4, 1, 1)
        mel = torch.nn.functional.conv2d(mel, weight)
        mel = mel.reshape(b, 2, 2, 40, dim).permute(0, 1, 3, 2, 4).reshape(b, 160, dim) * mask
        return mel, logdet


class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flows = torch.nn.ModuleList()
        for _ in range(config.DECODER_NUM_FLOWS):
            self.flows.append(Norm())
            self.flows.append(InvConvNear())
            self.flows.append(CouplingBlock())

    def forward(self, mel, mask, inference=False):
        lens = mel.shape[2]
        lens = (lens // 2) * 2
        mel = mel[:, :, :lens]
        mask = mask[:, :, :lens]
        mel = mel.reshape(-1, 80, lens // 2, 2).permute(0, 3, 1, 2).reshape(-1, 160, lens // 2)
        mask = mask[:, :, 1::2]
        mel = mel * mask
        logdet_sum = 0
        flows = reversed(self.flows) if inference else self.flows
        for i in flows:
            mel, logdet = i(mel, mask, inference=inference)
            logdet_sum += logdet
        lens = mel.shape[2]
        mel = mel.reshape(-1, 2, 80, lens).permute(0, 2, 3, 1).reshape(-1, 80, lens * 2)
        mask = mask.unsqueeze(3).repeat(1, 1, 1, 2).reshape(-1, 1, lens * 2)
        return mel * mask, logdet_sum


class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Conv1d(192, 192, 1)
        self.k = torch.nn.Conv1d(192, 192, 1)
        self.v = torch.nn.Conv1d(192, 192, 1)
        self.out = torch.nn.Conv1d(192, 192, 1)
        self.dropout = torch.nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)

    def forward(self, q, kv, mask):
        q, k, v = self.q(q), self.k(kv), self.v(kv)
        b, _, t = q.shape
        q = q.reshape(b, 2, 96, t).transpose(2, 3)
        k = k.reshape(b, 2, 96, t).transpose(2, 3)
        v = v.reshape(b, 2, 96, t).transpose(2, 3)
        atten = q.matmul(k.transpose(2, 3)) / 96 ** 0.5
        atten = atten.masked_fill(mask == 0, -1e4).softmax(dim=3)
        atten = self.dropout(atten).matmul(v)
        atten = atten.transpose(2, 3).reshape(b, 192, t)
        atten = self.out(atten)
        return atten


class EncoderNorm(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, c, 1) * 0.1)
        self.bias = torch.nn.Parameter(torch.zeros(1, c, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = ((x - mean) ** 2).mean(1, keepdim=True)
        x = (x - mean) / (var + 1e-4) ** 0.5
        return x * self.weight + self.bias


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Conv1d(192, 768, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.out = torch.nn.Conv1d(768, 192, 3, 1, 1)

    def forward(self, text, mask):
        text = self.s(text * mask)
        return self.out(text * mask) * mask


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = torch.nn.Embedding(config.PHONEME_VOCAB_SIZE, config.EMBEDDING_DIM)
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=config.EMBEDDING_DIM ** -0.5)
        self.input = torch.nn.Sequential(
            torch.nn.Conv1d(config.EMBEDDING_DIM, config.ENCODER_CONV_FILTERS, 5, padding=2),
            EncoderNorm(config.ENCODER_CONV_FILTERS), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Conv1d(config.ENCODER_CONV_FILTERS, config.ENCODER_CONV_FILTERS, 5, padding=2),
            EncoderNorm(config.ENCODER_CONV_FILTERS), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Conv1d(config.ENCODER_CONV_FILTERS, config.ENCODER_CONV_FILTERS, 5, padding=2),
            EncoderNorm(config.ENCODER_CONV_FILTERS), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Conv1d(config.ENCODER_CONV_FILTERS, config.ENCODER_CONV_FILTERS, 1),
        )
        self.input[-1].weight.data.zero_()
        self.input[-1].bias.data.zero_()
        self.atten = torch.nn.ModuleList([Atten() for _ in range(config.ENCODER_NUM_LAYERS)])
        self.norm1 = torch.nn.ModuleList([EncoderNorm(config.EMBEDDING_DIM) for _ in range(config.ENCODER_NUM_LAYERS)])
        self.ffn = torch.nn.ModuleList([FFN() for _ in range(config.ENCODER_NUM_LAYERS)])
        self.norm2 = torch.nn.ModuleList([EncoderNorm(config.EMBEDDING_DIM) for _ in range(config.ENCODER_NUM_LAYERS)])
        self.dropout = torch.nn.Dropout(0.1)
        self.out1 = torch.nn.Conv1d(config.EMBEDDING_DIM, 80, 1)
        self.out2_1 = torch.nn.Sequential(
            torch.nn.Conv1d(config.EMBEDDING_DIM, 256, 3, padding=1),
            torch.nn.ReLU(), EncoderNorm(256), torch.nn.Dropout(0.1),
        )
        self.out2_2 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 256, 3, padding=1),
            torch.nn.ReLU(), EncoderNorm(256), torch.nn.Dropout(0.1),
        )
        self.out2_3 = torch.nn.Conv1d(256, 1, 1)

    def forward(self, text, mask):
        text = self.embed(text) * 192 ** 0.5
        text = text.transpose(1, 2)
        text = (self.input(text) + text) * mask
        mask_atten = mask.unsqueeze(2) * mask.unsqueeze(3)
        for i in range(self.config.ENCODER_NUM_LAYERS):
            text = text * mask
            atten = self.atten[i](text, text, mask_atten)
            atten = self.dropout(atten)
            text = self.norm1[i](text + atten)
            atten = self.ffn[i](text, mask)
            atten = self.dropout(atten)
            text = self.norm2[i](text + atten)
        text = text * mask
        out1 = self.out1(text) * mask
        text = self.out2_1(text.detach() * mask)
        text = self.out2_2(text * mask)
        out2 = self.out2_3(text * mask) * mask
        return out1, out2


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, text, lens_text, mel, lens_mel):
        lens_mel_sum = lens_mel.sum()
        mask_text = get_mask(lens_text).unsqueeze(1).to(torch.float32)
        encode1, encode2 = self.encoder(text, mask_text)
        lens_mel = lens_mel // 2 * 2
        mel = mel[:, :, :max(lens_mel)]
        mask_mel = get_mask(lens_mel).unsqueeze(1).to(mask_text.dtype)
        mask_atten = mask_text.unsqueeze(3) * mask_mel.unsqueeze(2)
        mask_atten = mask_atten.squeeze(1)
        mel, logdet = self.decoder(mel, mask_mel, inference=False)
        with torch.no_grad():
            logp1 = torch.full_like(encode1, -0.5 * math.log(2 * math.pi)).sum(1).unsqueeze(2)
            logp2 = (-0.5 * mel ** 2).sum(1, keepdim=True).repeat(1, encode1.shape[2], 1)
            logp3 = encode1.transpose(1, 2).matmul(mel)
            logp4 = (-0.5 * (encode1 ** 2)).sum(1).unsqueeze(2)
            logp = logp1 + logp2 + logp3 + logp4
            logp = maximum_path(logp, mask_atten)
        out1 = logp.transpose(1, 2).matmul(encode1.transpose(1, 2)).transpose(1, 2)
        out2 = (1 + logp.sum(2).unsqueeze(1)).log() * mask_text
        mel = mel.transpose(1, 2)
        out1 = out1.transpose(1, 2)
        encode2 = encode2.transpose(1, 2)
        out2 = out2.transpose(1, 2)
        r = 0.5 * math.log(2 * math.pi)
        loss_recon = 0.5 * ((mel - out1) ** 2).sum()
        loss_recon = r + (loss_recon - logdet.sum()) / (lens_mel_sum * mel.shape[2])
        loss_dur = ((encode2 - out2) ** 2).sum() / lens_text.sum()
        total_loss = self.config.LOSS_W_RECON * loss_recon + self.config.LOSS_W_DUR * loss_dur
        return total_loss

    @torch.no_grad()
    def inference(self, text):
        lens_text = torch.LongTensor([text.shape[1]])
        mask_text = get_mask(lens_text).unsqueeze(1).float()
        encode1, encode2 = self.encoder(text, mask_text)
        encode2 = (encode2.exp() - 1) * mask_text
        encode2 = encode2.ceil().clamp_min(1).long()
        lens_mel = encode2.sum([1, 2]).clamp_min(1)
        mask_mel = get_mask(lens_mel).unsqueeze(1).float()
        mask_atten = (mask_text.unsqueeze(3) * mask_mel.unsqueeze(2)).squeeze(1)
        atten = encode2.cumsum(2).reshape(lens_text)
        atten = get_mask(atten).float()
        for i in range(lens_text.item() - 1, 0, -1):
            atten[i] = atten[i] - atten[i - 1]
        atten = (atten.unsqueeze(0) * mask_atten).transpose(1, 2)
        mel = atten.matmul(encode1.transpose(1, 2)).transpose(1, 2) * mask_mel
        mel, _ = self.decoder(mel, mask_mel, inference=True)
        mel = mel.transpose(1, 2)
        return mel


def generate_training_analysis(train_losses, val_losses, val_accuracies, grad_norms, epoch_times,
                               config, model_name, plot_dir, best_val_loss):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='训练损失')
    plt.plot(epochs, val_losses, 'r-s', label='验证损失')
    best_epoch = np.argmin(val_losses) + 1
    plt.scatter(best_epoch, best_val_loss, color='green', s=100, zorder=5,
                label=f'最佳验证损失 (第{best_epoch}轮)')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title(f'损失曲线 - {model_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    window = min(5, len(train_losses))
    if window > 1:
        train_smooth = np.convolve(train_losses, np.ones(window) / window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window) / window, mode='valid')
        smooth_epochs = epochs[window - 1:]
        plt.subplot(1, 2, 2)
        plt.plot(smooth_epochs, train_smooth, 'b-', label='训练损失(平滑)')
        plt.plot(smooth_epochs, val_smooth, 'r-', label='验证损失(平滑)')
        plt.xlabel('训练轮数')
        plt.ylabel('平滑损失值')
        plt.title('损失平滑曲线 (移动平均)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'loss_analysis_{model_name}.png'), dpi=200)
    plt.close()

    val_diff = np.diff(val_losses)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs[1:], val_diff, 'g-o', markersize=4)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('训练轮数')
    plt.ylabel('验证损失下降率')
    plt.title('收敛速度分析')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, f'loss_decrease_rate_{model_name}.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, grad_norms, 'purple', marker='^', markersize=4)
    plt.xlabel('训练轮数')
    plt.ylabel('平均梯度范数')
    plt.title('梯度稳定性分析')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, f'grad_norm_{model_name}.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_accuracies, 'teal', marker='o', markersize=4)
    plt.xlabel('训练轮数')
    plt.ylabel('验证准确率 (1 - val_loss / initial_val_loss)')
    plt.title('验证准确率曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, f'accuracy_{model_name}.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(epochs, epoch_times, color='steelblue', alpha=0.7)
    plt.xlabel('训练轮数')
    plt.ylabel('时间 (秒)')
    plt.title('每轮训练耗时')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, f'epoch_time_{model_name}.png'), dpi=200)
    plt.close()

    last_10_val = val_losses[-10:]
    std_last10 = np.std(last_10_val)
    mean_last10 = np.mean(last_10_val)
    min_val = min(val_losses)
    min_epoch = np.argmin(val_losses) + 1
    drop_percent = (val_losses[0] - min_val) / val_losses[0] * 100
    overfit_ratio = (val_losses[-1] - min_val) / (min_val + 1e-8)
    overfit_warning = "是" if overfit_ratio > 0.05 else "否"
    if std_last10 < 0.01 * mean_last10:
        convergence_status = "已收敛（最后10个epoch损失稳定）"
    elif drop_percent > 50:
        convergence_status = "仍在下降，可继续训练"
    else:
        convergence_status = "收敛缓慢，可考虑早停"

    report = f"""
========================================
    模型训练收敛性分析报告
模型名称: {model_name}
最佳验证损失: {min_val:.6f} (第{min_epoch}轮)
最终验证准确率: {val_accuracies[-1]:.4f}
过拟合风险: {overfit_warning}
收敛性评估: {convergence_status}
========================================
"""
    report_path = os.path.join(plot_dir, f'convergence_report_{model_name}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)


def train_model(config, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[训练] 使用设备: {device}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(config.OUTPUT_DIR, 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)

    train_dataset = TTSDataset(config, mode='train')
    vocab_path = os.path.join(config.VOCAB_DIR, 'pinyin_vocab.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            config.PHONEME_VOCAB_SIZE = len(vocab)

    model = Model(config)
    model.train()
    model.to(device)

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

    def warmup_then_cosine(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_then_cosine(epoch, config.WARMUP_STEPS, config.NUM_EPOCHS)
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    train_loader = get_loader(config, mode='train')
    val_loader = get_loader(config, mode='val')

    best_val_loss = float('inf')
    early_stop_counter = 0
    model_save_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    best_model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pth")

    train_loss_history, val_loss_history = [], []
    val_accuracy_history, grad_norm_history, epoch_time_history = [], [], []
    initial_val_loss = None

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm_sum = 0.0
        num_batches = 0
        for batch_data in tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{config.NUM_EPOCHS}]'):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                loss = model(**batch_data)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[严重警告] 损失非法 (NaN/Inf)，跳过批次")
                optimizer.zero_grad()
                if device == 'cuda': torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            grad_norm_val = grad_norm.item()

            if math.isnan(grad_norm_val) or math.isinf(grad_norm_val):
                print(f"[严重警告] 梯度非法 (NaN/Inf)，跳过批次")
                optimizer.zero_grad()
                scaler.update()
                if device == 'cuda': torch.cuda.empty_cache()
                continue

            epoch_grad_norm_sum += grad_norm_val
            num_batches += 1
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if num_batches == 0:
            return None

        train_avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm_sum / num_batches
        train_loss_history.append(train_avg_loss)
        grad_norm_history.append(avg_grad_norm)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                for k in batch_data:
                    batch_data[k] = batch_data[k].to(device, non_blocking=True)
                loss = model(**batch_data)
                val_loss += loss.item()
        val_avg_loss = val_loss / len(val_loader)
        val_loss_history.append(val_avg_loss)

        if initial_val_loss is None:
            initial_val_loss = val_avg_loss
            val_accuracy = 0.0
        else:
            val_accuracy = max(0.0, 1.0 - val_avg_loss / initial_val_loss)
        val_accuracy_history.append(val_accuracy)
        epoch_time_history.append(time.time() - epoch_start_time)

        print(f"Epoch [{epoch + 1}] 训练损失: {train_avg_loss:.6f}, 验证损失: {val_avg_loss:.6f}, 准确率: {val_accuracy:.4f}")

        if val_avg_loss < best_val_loss - config.EARLY_STOP_MIN_DELTA:
            best_val_loss = val_avg_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"[早停] 验证损失未改善，终止训练。")
            break
        scheduler.step()

    torch.save(model.state_dict(), model_save_path)
    generate_training_analysis(train_loss_history, val_loss_history, val_accuracy_history,
                               grad_norm_history, epoch_time_history, config, model_name, plot_dir, best_val_loss)

    package_dir = config.get_model_package_path(model_name)
    shutil.copy2(best_model_path, os.path.join(package_dir, f"{model_name}_best.pth"))
    shutil.copy2(model_save_path, os.path.join(package_dir, f"{model_name}.pth"))
    shutil.copy2(os.path.join(config.VOCAB_DIR, 'pinyin_vocab.pkl'), os.path.join(package_dir, 'vocab.pkl'))

    with open(os.path.join(package_dir, 'vocab.pkl'), 'rb') as f:
        real_vocab_size = len(pickle.load(f))
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
        'Final_Train_Loss': train_loss_history[-1] if train_loss_history else None,
        'Final_Val_Loss': val_loss_history[-1] if val_loss_history else None,
    }
    with open(os.path.join(package_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config_to_save, f)
    return package_dir


def load_model_package(package_dir):
    package_dir = Path(package_dir)
    if not package_dir.is_dir():
        raise FileNotFoundError(f"模型包目录不存在: {package_dir}")
    config_path = package_dir / 'config.pkl'
    if not config_path.exists():
        raise FileNotFoundError(f"缺少配置文件")
    with open(config_path, 'rb') as f:
        saved_config = pickle.load(f)
    config = TTSConfig()
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    print(f"[加载] 配置已恢复，采样率={config.SAMPLE_RATE}, N_MELS={config.N_MELS}")

    vocab_path = package_dir / 'vocab.pkl'
    if not vocab_path.exists():
        raise FileNotFoundError(f"缺少词汇表文件")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = ChineseTokenizer(config)
    tokenizer.vocab = vocab
    tokenizer.id_to_token = {v: k for k, v in vocab.items()}
    tokenizer.next_id = max(vocab.values()) + 1 if vocab else 106
    print(f"[加载] 词汇表大小: {len(vocab)}")

    model_files = list(package_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"未找到模型文件")
    best_model = None
    for m in model_files:
        if '_best' in m.stem:
            best_model = m
            break
    model_path = best_model if best_model else model_files[0]
    print(f"[加载] 正在加载模型: {model_path.name}")
    state_dict = torch.load(model_path, map_location='cpu')
    config.PHONEME_VOCAB_SIZE = state_dict['encoder.embed.weight'].shape[0]
    model = Model(config)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[加载] 模型加载完成")
    return config, tokenizer, model


def synthesize_speech_from_package(package_dir, text, save_mel=True, volume=None, pitch_shift=None, speed_rate=None):
    config, tokenizer, model = load_model_package(package_dir)

    model_name = os.path.basename(package_dir)
    timestamp = int(time.time())
    output_dir = os.path.join(config.OUTPUT_DIR, model_name, str(timestamp))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[合成] 输入文本: {text}")
    token_ids = tokenizer(text, update_vocab=False)
    if not token_ids:
        raise ValueError("Token序列为空，请检查输入文本")
    print(f"[合成] Token序列长度: {len(token_ids)}")
    print(f"[合成] 解码还原: {tokenizer.decode(token_ids)}")

    token_ids = [tokenizer.vocab['<BOS>']] + token_ids + [tokenizer.vocab['<EOS>']]
    text_tensor = torch.LongTensor(token_ids).unsqueeze(0)

    with torch.no_grad():
        mel = model.inference(text_tensor)
    mel_np = mel[0].numpy().T
    print(f"[合成] 梅尔谱形状: {mel_np.shape}")
    print(f"[合成] 梅尔谱范围: [{mel_np.min():.4f}, {mel_np.max():.4f}], 均值: {mel_np.mean():.4f}")

    processor = ChineseProcessor(config)
    if save_mel:
        mel_path = os.path.join(output_dir, f"mel_{timestamp}.png")
        processor.save_mel_spectrogram(mel_np, mel_path, title=f"合成: {text[:30]}...")

    wav = processor.mel_to_wav(mel_np, volume=volume, pitch_shift=pitch_shift, speed_rate=speed_rate)
    wav = processor.denoise_wav(wav)

    if wav.dtype == np.int16:
        wav_float = wav.astype(np.float32) / 32768.0
    else:
        wav_float = wav.astype(np.float32)
    peak = np.max(np.abs(wav_float))
    if peak > 1e-8:
        wav_float = wav_float * (0.95 / peak)
    wav_int16 = (wav_float * 32767).clip(-32768, 32767).astype(np.int16)

    wav_path = os.path.join(output_dir, f"output_{timestamp}.wav")
    scipy.io.wavfile.write(wav_path, config.SAMPLE_RATE, wav_int16)
    print(f"[合成] 音频已保存至: {wav_path}")
    return wav_path, mel_path


# ===================== 交互式菜单 =====================
def list_model_packages(config):
    os.makedirs(config.MODEL_PACKAGE_DIR, exist_ok=True)
    package_dirs = [d for d in os.listdir(config.MODEL_PACKAGE_DIR)
                    if os.path.isdir(os.path.join(config.MODEL_PACKAGE_DIR, d))]
    if not package_dirs:
        print("[提示] 未找到任何模型包。")
        return [], None
    print("\n=== 可用模型包列表 ===")
    for i, p in enumerate(package_dirs):
        p_path = os.path.join(config.MODEL_PACKAGE_DIR, p)
        model_files = glob(os.path.join(p_path, "*.pth"))
        if model_files:
            print(f"{i+1}. [{p}] - {len(model_files)} 个模型文件")
        else:
            print(f"{i+1}. [{p}] - (无模型文件)")
    return package_dirs, len(package_dirs)


def get_user_choice(prompt, valid_range=None, default=None):
    while True:
        inp = input(prompt).strip()
        if not inp and default is not None:
            return default
        try:
            choice = int(inp)
            if valid_range and choice not in valid_range:
                print(f"[错误] 请输入 {valid_range} 内的数字")
                continue
            return choice
        except ValueError:
            print("[错误] 请输入数字")


def clear_cache(config):
    for d in [config.CACHE_TEXT_PATH, config.CACHE_MEL_PATH]:
        if os.path.exists(d):
            for f in glob(os.path.join(d, "*cache*.pkl")):
                os.remove(f)
    print("[缓存] 已清理文本/梅尔缓存。")


def main_menu():
    config = TTSConfig()
    while True:
        print("\n=== 中文TTS系统 ===")
        print("1. TTS模型训练")
        print("2. 语音合成")
        print("0. 退出")
        choice = get_user_choice("请选择 (0-2): ", range(0,3))
        if choice == 1:
            train_interactive(config)
        elif choice == 2:
            synthesize_interactive(config)
        else:
            print("退出。")
            exit(0)


def train_interactive(config):
    print("\n=== 模型训练 ===")
    choice = get_user_choice("1. 数据集 (0=默认, 1=自定义, 默认0): ", [0,1], 0)
    if choice == 1:
        p = input("输入数据集路径: ").strip()
        if os.path.exists(p):
            config.DATASET_PATH = p
        else:
            print("路径不存在，使用默认。")
    choice = get_user_choice("2. 调整训练参数? (0=默认, 1=自定义, 默认0): ", [0,1], 0)
    if choice == 1:
        config = adjust_training_params(config)
    model_name = input("3. 模型名称: ").strip()
    if not model_name:
        model_name = "default"
    if input(f"确认开始训练 {model_name}? (y/n): ").strip().lower() != 'n':
        package_path = train_model(config, model_name)
        print(f"训练完成，模型包: {package_path}")
    input("按回车返回...")


def adjust_training_params(config):
    print("\n=== 调整训练参数 (直接回车使用默认值) ===")
    bs = input(f"批次大小 ({config.BATCH_SIZE}): ").strip()
    if bs: config.BATCH_SIZE = int(bs)
    lr = input(f"学习率 ({config.LEARNING_RATE}): ").strip()
    if lr: config.LEARNING_RATE = float(lr)
    ep = input(f"训练轮数 ({config.NUM_EPOCHS}): ").strip()
    if ep: config.NUM_EPOCHS = int(ep)
    wu = input(f"预热轮数 ({config.WARMUP_STEPS}): ").strip()
    if wu: config.WARMUP_STEPS = int(wu)
    print("参数已更新。")
    return config


def synthesize_interactive(config):
    print("\n=== 语音合成 ===")
    packages, count = list_model_packages(config)
    if not packages:
        return
    choice = get_user_choice(f"选择模型包 (1-{count}): ", range(1, count+1))
    pkg_path = os.path.join(config.MODEL_PACKAGE_DIR, packages[choice-1])
    txt_choice = get_user_choice("文本输入 (0=手动, 1=文件, 默认0): ", [0,1], 0)
    if txt_choice == 0:
        text = input(f"输入文本 (默认: {config.INFERENCE_TEXT}): ").strip()
        if not text:
            text = config.INFERENCE_TEXT
    else:
        fpath = input("txt文件路径: ").strip()
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    param_choice = get_user_choice("调整语音参数? (0=默认, 1=自定义, 默认0): ", [0,1], 0)
    vol, pitch, speed = config.VOLUME, config.PITCH_SHIFT, config.SPEED_RATE
    if param_choice == 1:
        v = input(f"音量 ({vol}): ").strip()
        if v: vol = max(0.1, min(5.0, float(v)))
        p = input(f"音调 ({pitch}): ").strip()
        if p: pitch = max(-12, min(12, int(p)))
        s = input(f"语速 ({speed}): ").strip()
        if s: speed = max(0.5, min(2.0, float(s)))
    wav_p, mel_p = synthesize_speech_from_package(pkg_path, text, save_mel=True,
                                                volume=vol, pitch_shift=pitch, speed_rate=speed)
    print(f"合成完成！音频: {wav_p}")
    input("按回车返回...")


if __name__ == '__main__':
    c = TTSConfig()
    os.makedirs(c.ROOT_DIR, exist_ok=True)
    os.makedirs(c.MODEL_DIR, exist_ok=True)
    os.makedirs(c.OUTPUT_DIR, exist_ok=True)
    os.makedirs(c.MODEL_PACKAGE_DIR, exist_ok=True)
    main_menu()