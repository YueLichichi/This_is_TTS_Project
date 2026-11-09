# 这是TTS - 中文语音合成系统

> ⚠️ **本项目仅供学习与研究使用，请勿用于任何商业用途。**

基于 **Glow-TTS** 架构的中文语音合成（TTS）系统，使用 Python + PyTorch + Flask 构建，覆盖从 **数据 → 训练 → 合成 → 评测** 的完整闭环，并提供开箱即用的 Web 界面。

## ✨ 项目特点

- **端到端闭环**：数据预处理、模型训练、语音合成、客观评测一站式完成，无需切换多个工具
- **自定义音色**：只需准备「wav + 同名 txt」格式的数据集（递归扫描任意目录层级），即可训练专属音色
- **实时训练可视化**：Web 页面实时展示每个 Epoch / Batch 的 Loss、准确率、梯度范数，自动生成训练曲线图与收敛报告；支持随时启动 / 停止、早停保护
- **灵活的合成控制**：输入中文文本即可生成语音，支持音量、音调（半音）、语速三维调节，同时输出梅尔频谱图
- **客观质量评测**：上传参考音频，一键计算 MCD（梅尔倒谱失真）与 STOI（短时客观可懂度）指标
- **模型自包含打包**：训练完成自动将权重 + 词汇表 + 配置打包为独立模型包，迁移、分发、加载零配置
- **稳定训练保障**：混合精度训练、梯度裁剪、梯度爆炸自动跳批、余弦退火学习率、早停机制
- **开箱即用**：内置 ffmpeg，克隆仓库后装好依赖即可运行

## 目录结构

```
TTS_Project/
├── backend/
│   ├── app.py                    # Flask 后端（API 服务 + 训练调度）
│   └── CN_CustomTone_GlowTTS.py  # 核心框架（模型、数据处理、训练、合成）
├── frontend/
│   ├── index.html                # Web 前端页面
│   ├── css/ / js/app.js          # 样式与前端逻辑
├── ffmpeg/                       # ffmpeg 目录（bin/ 下的可执行文件不随仓库分发，请自行下载放入）
├── venv/                         # Python 虚拟环境
└── README.md
```

## 快速开始

```powershell
# 1. 安装依赖
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch numpy librosa soundfile scipy pypinyin matplotlib tqdm flask Flask-CORS

# 2. 启动服务
cd backend
python app.py
```

访问 <http://localhost:5000>，按「准备数据集 → 训练 → 合成 → 评测」流程使用。

### 数据集格式

任意目录层级下放置「音频-文本对」，训练时填入数据集根目录路径即可：

```
dataset/
├── chapter1_1_player_001.wav
├── chapter1_1_player_001.txt     # 与 wav 同名，UTF-8 编码
└── ...
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/list_models` | 列出可用模型包 |
| POST | `/api/train` | 启动训练（`model_name` / `dataset_path` / `epochs` / `batch_size` / `lr`） |
| GET  | `/api/train_status` | 查询训练实时状态与日志 |
| POST | `/api/stop_train` | 停止当前训练 |
| POST | `/api/synthesize` | 语音合成（`text` / `volume` / `pitch` / `speed`） |
| POST | `/api/upload_ref` | 上传评测参考音频 |
| POST | `/api/evaluate` | MCD / STOI 客观评测 |

## 免责声明

- 本项目基于开源论文与框架学习实现，**仅供个人学习、课程设计与学术研究使用**
- 请勿将训练或合成的音色用于冒充他人、诈骗、传播虚假信息等违法用途
- 请使用您拥有合法权利的音频数据训练模型，使用本项目即表示您已知晓并同意上述条款
