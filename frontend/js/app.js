// frontend/js/app.js
let trainingPollingInterval = null;
let lossChart = null;
let currentTrainLosses = [];
let currentValLosses = [];

const API_BASE = '';

// 辅助函数
function showToast(message, isError = false) {
    const toast = document.createElement('div');
    toast.className = `toast-message ${isError ? 'error-toast' : ''}`;
    toast.innerHTML = `<i class="fas ${isError ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i> ${message}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// 损失图表
function initLossChart(trainData, valData) {
    const ctx = document.getElementById('lossChart').getContext('2d');
    if (lossChart) lossChart.destroy();
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: Math.max(trainData.length, 1) }, (_, i) => i + 1),
            datasets: [
                { label: '训练损失', data: trainData, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)', tension: 0.2, fill: true },
                { label: '验证损失', data: valData, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.05)', tension: 0.2, fill: true }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: true,
            plugins: { legend: { position: 'top' }, tooltip: { mode: 'index' } },
            scales: { y: { title: { display: true, text: '损失值' } }, x: { title: { display: true, text: '轮次' } } }
        }
    });
}

function updateLossChart(trainLosses, valLosses) {
    if (lossChart) {
        lossChart.data.datasets[0].data = trainLosses;
        lossChart.data.datasets[1].data = valLosses;
        lossChart.options.scales.x.max = Math.max(trainLosses.length, 1);
        lossChart.update();
    } else {
        initLossChart(trainLosses, valLosses);
    }
}

// 训练状态轮询
async function fetchTrainingStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/train_status`);
        const status = await res.json();
        if (!status) return;

        const epochPercent = status.total_epochs ? (status.progress / status.total_epochs) * 100 : 0;
        document.getElementById('trainProgressFill').style.width = `${epochPercent}%`;
        document.getElementById('epochStatus').innerText = `${status.progress} / ${status.total_epochs}`;

        if (status.running || status.current_epoch > 0) {
            document.getElementById('tqdmText').innerHTML = '🚀 训练中 ...';
            document.getElementById('tqdmEpoch').innerHTML = `Epoch ${status.current_epoch || 0}/${status.total_epochs || 0}`;
            document.getElementById('tqdmBatch').innerHTML = `Batch ${status.current_batch || 0}/${status.total_batches || 0}`;
            document.getElementById('tqdmLoss').innerHTML = `loss: ${(status.current_loss || 0).toFixed(6)}`;
            const batchProgress = status.batch_progress || 0;
            document.getElementById('tqdmProgressFill').style.width = `${batchProgress}%`;
            document.getElementById('tqdmProgressFill').style.backgroundColor = batchProgress === 100 ? '#10b981' : '#2563eb';
        } else if (!status.running && status.total_epochs > 0) {
            document.getElementById('tqdmText').innerHTML = '✅ 训练完成！';
            document.getElementById('tqdmProgressFill').style.width = '100%';
        } else {
            document.getElementById('tqdmText').innerHTML = '等待训练启动...';
            document.getElementById('tqdmProgressFill').style.width = '0%';
        }

        if (status.log && status.log.length) {
            const logDiv = document.getElementById('trainLogBox');
            logDiv.innerHTML = status.log.slice(-30).map(l => `<div class="log-line">📌 ${l}</div>`).join('');
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        if (status.train_losses && status.val_losses) {
            currentTrainLosses = status.train_losses;
            currentValLosses = status.val_losses;
            updateLossChart(currentTrainLosses, currentValLosses);
        }

        if (!status.running && trainingPollingInterval) {
            clearInterval(trainingPollingInterval);
            trainingPollingInterval = null;
            document.getElementById('startTrainBtn').disabled = false;
            document.getElementById('stopTrainBtn').disabled = true;
            showToast('✅ 训练已完成！模型已保存至 model_packages', false);
            if (status.analysis_images && status.analysis_images.length) {
                const analysisDiv = document.getElementById('analysisArea');
                const imagesContainer = document.getElementById('analysisImages');
                imagesContainer.innerHTML = '';
                status.analysis_images.forEach(url => {
                    if (url.endsWith('.png')) {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.style.padding = '12px';
                        card.innerHTML = `<img src="${url}" style="width:100%; border-radius:12px;" alt="分析图表">`;
                        imagesContainer.appendChild(card);
                    } else if (url.endsWith('.txt')) {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.style.padding = '16px';
                        card.innerHTML = `<a href="${url}" target="_blank" style="text-decoration:none;"><i class="fas fa-file-alt"></i> 下载收敛性报告</a>`;
                        imagesContainer.appendChild(card);
                    }
                });
                analysisDiv.style.display = 'block';
            }
        }
    } catch (err) {
        console.error('轮询失败', err);
    }
}

async function startTraining(data) {
    try {
        const res = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || '启动失败');
        }
        showToast('训练任务已启动，请等待进度更新');
        document.getElementById('startTrainBtn').disabled = true;
        document.getElementById('stopTrainBtn').disabled = false;
        if (trainingPollingInterval) clearInterval(trainingPollingInterval);
        trainingPollingInterval = setInterval(fetchTrainingStatus, 1000);
        fetchTrainingStatus();
    } catch (err) {
        showToast(err.message, true);
        document.getElementById('startTrainBtn').disabled = false;
    }
}

// 语音合成
async function synthesizeSpeech(data) {
    const synthBtn = document.getElementById('synthBtn');
    const originalText = synthBtn.innerHTML;
    synthBtn.disabled = true;
    synthBtn.innerHTML = '<span class="spinner"></span> 合成中...';
    const resultDiv = document.getElementById('synthResultArea');
    try {
        const res = await fetch(`${API_BASE}/api/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || '合成失败');
        }
        const result = await res.json();
        // 保存本地路径供评测使用
        window.lastSynAudioPath = result.local_path;
        // 构建音频播放器和梅尔图
        let audioHtml = `<div class="audio-player"><i class="fas fa-play-circle" style="font-size:1.6rem;"></i><audio controls src="${result.audio_url}" style="width:100%;">您的浏览器不支持音频播放</audio></div>`;
        let melHtml = '';
        if (result.mel_url) {
            melHtml = `<img class="mel-img" src="${result.mel_url}" alt="梅尔频谱图" style="max-width:100%; border-radius:16px;">`;
        } else {
            melHtml = `<div class="log-box" style="margin-top:12px;"><i class="fas fa-chart-simple"></i> 未返回频谱图，请检查后端</div>`;
        }
        // 评测区域
        const evalArea = `<div id="evalArea" style="margin-top: 16px;">
            <button id="evalBtn" class="secondary"><i class="fas fa-chart-line"></i> 客观评测 (MCD+STOI)</button>
            <div id="evalResult" style="margin-top: 12px; font-family: monospace;"></div>
        </div>`;
        resultDiv.innerHTML = audioHtml + melHtml + evalArea;
        // 绑定评测按钮事件
        document.getElementById('evalBtn').addEventListener('click', () => startEvaluation());
        showToast('✅ 合成成功，可播放语音');
    } catch (err) {
        resultDiv.innerHTML = `<div style="color:#dc2626; padding:16px; background:#fee2e2; border-radius:20px;"><i class="fas fa-exclamation-circle"></i> 合成错误: ${err.message}</div>`;
        showToast(err.message, true);
    } finally {
        synthBtn.disabled = false;
        synthBtn.innerHTML = originalText;
    }
}

// 客观评测流程
async function startEvaluation() {
    const synPath = window.lastSynAudioPath;
    if (!synPath) {
        showToast('没有可评测的合成音频', true);
        return;
    }
    // 弹出文件选择框，要求用户上传参考音频（原始录音）
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'audio/wav';
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        // 显示加载状态
        const evalResultDiv = document.getElementById('evalResult');
        evalResultDiv.innerHTML = '<div class="spinner" style="margin: 8px auto;"></div><div>正在评测，请稍候...</div>';
        try {
            // 1. 上传参考音频
            const formData = new FormData();
            formData.append('file', file);
            const uploadRes = await fetch(`${API_BASE}/api/upload_ref`, {
                method: 'POST',
                body: formData
            });
            if (!uploadRes.ok) {
                const err = await uploadRes.json();
                throw new Error(err.error || '上传失败');
            }
            const { temp_path } = await uploadRes.json();
            // 2. 调用评测接口
            const evalRes = await fetch(`${API_BASE}/api/evaluate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    syn_audio_path: synPath,
                    ref_audio_path: temp_path
                })
            });
            if (!evalRes.ok) {
                const err = await evalRes.json();
                throw new Error(err.error || '评测失败');
            }
            const evalData = await evalRes.json();
            // 显示结果
            let stoiText = evalData.stoi !== null ? evalData.stoi.toFixed(4) : '未安装pystoi';
            evalResultDiv.innerHTML = `
                <div style="background:#eef2ff; padding:12px; border-radius:16px;">
                    <strong>📊 客观评测结果</strong><br>
                    MCD (梅尔倒谱失真): <strong>${evalData.mcd}</strong> (越小越好, 通常<10为优)<br>
                    STOI (短时可懂度): <strong>${stoiText}</strong> (越接近1越可懂)
                </div>
            `;
            showToast('评测完成');
        } catch (err) {
            evalResultDiv.innerHTML = `<div style="color:#dc2626;">评测失败: ${err.message}</div>`;
            showToast(err.message, true);
        }
    };
    fileInput.click();
}

// 加载模型包列表
async function loadModelPackages() {
    try {
        const res = await fetch(`${API_BASE}/api/list_models`);
        const data = await res.json();
        const select = document.getElementById('modelSelect');
        if (data.models && data.models.length) {
            select.innerHTML = '<option value="">-- 请选择模型 --</option>' + data.models.map(m => `<option value="${m}">${m}</option>`).join('');
        } else {
            select.innerHTML = '<option value="">暂无模型，请先训练</option>';
        }
    } catch (e) {
        console.warn('获取模型列表失败', e);
        document.getElementById('modelSelect').innerHTML = '<option value="">无法连接后端</option>';
    }
}

function setupTxtUpload() {
    const uploadBtn = document.getElementById('uploadTxtBtn');
    const fileInput = document.getElementById('txtFileInput');
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            document.querySelector('textarea[name="text"]').value = ev.target.result;
            showToast('已加载文本文件');
        };
        reader.readAsText(file, 'UTF-8');
    });
}

function setupStopTrain() {
    document.getElementById('stopTrainBtn').addEventListener('click', () => {
        showToast('后端未实现停止训练接口，可通过重启服务终止', true);
    });
}

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (tabId === 'synth-tab') loadModelPackages();
        });
    });
}

function bindFormEvents() {
    document.getElementById('trainForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = {
            model_name: formData.get('model_name'),
            dataset_path: formData.get('dataset_path') || '',
            epochs: parseInt(formData.get('epochs')),
            batch_size: parseInt(formData.get('batch_size')),
            lr: parseFloat(formData.get('lr'))
        };
        if (!data.model_name) { showToast('请输入模型名称', true); return; }
        await startTraining(data);
    });

    document.getElementById('synthForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const modelName = document.getElementById('modelSelect').value;
        if (!modelName) { showToast('请选择一个模型包', true); return; }
        const formData = new FormData(e.target);
        const data = {
            model_name: modelName,
            text: formData.get('text'),
            volume: parseFloat(formData.get('volume')),
            pitch: parseInt(formData.get('pitch')),
            speed: parseFloat(formData.get('speed'))
        };
        if (!data.text.trim()) { showToast('请输入合成文本', true); return; }
        await synthesizeSpeech(data);
    });
}

function init() {
    setupTabs();
    bindFormEvents();
    setupTxtUpload();
    setupStopTrain();
    loadModelPackages();
    fetchTrainingStatus();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}