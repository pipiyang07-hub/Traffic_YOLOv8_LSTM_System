# 基于YOLOv8与LSTM的交通目标检测与流量预测系统

## 项目简介

本系统是一个基于深度学习的交通流量智能分析平台，实现了以下功能：

- **交通目标检测**：使用YOLOv8对交通视频中的车辆进行实时检测
- **流量统计**：采用虚拟检测线法对过往车辆进行分类计数
- **流量预测**：基于LSTM时间序列模型预测未来交通流量
- **可视化展示**：通过Streamlit构建交互式Web界面

## 项目结构

```
Traffic_YOLOv8_LSTM_System/
├── configs/                    # 配置文件
│   └── config.yaml            # 主配置文件
│
├── src/                       # 源代码
│   ├── detection/             # 目标检测模块
│   │   └── detector.py        # YOLOv8检测器
│   │
│   ├── counting/              # 流量统计模块
│   │   ├── line.py            # 检测线定义
│   │   └── counter.py         # 车辆计数
│   │
│   ├── prediction/            # 流量预测模块
│   │   ├── lstm_model.py      # LSTM模型
│   │   ├── trainer.py         # 模型训练
│   │   └── predictor.py       # 流量预测
│   │
│   ├── visualization/         # 可视化模块
│   │   ├── plots.py           # 图表绘制
│   │   └── app.py             # Streamlit应用
│   │
│   └── utils/                 # 工具函数
│       ├── logger.py          # 日志工具
│       ├── video_utils.py     # 视频处理
│       └── data_utils.py      # 数据处理
│
├── data/                      # 数据目录
│   ├── videos/                # 输入视频
│   ├── output/                # 输出结果
│   ├── models/                # 保存的模型
│   └── traffic_data.csv       # 流量统计数据
│
├── main.py                    # 主入口（检测+计数）
├── train.py                   # 模型训练入口
├── requirements.txt           # 依赖清单
└── README.md                  # 项目说明
```

## 安装

### 1. 创建虚拟环境（推荐）

```bash
conda create -n traffic python=3.10
conda activate traffic
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载YOLOv8模型

首次运行时，程序会自动下载YOLOv8模型。也可以手动下载：

```bash
# 下载yolov8n模型（推荐，速度快）
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
```

## 使用方法

### 1. 视频检测与计数

```bash
# 基本使用
python main.py --video data/videos/traffic.mp4

# 自定义参数
python main.py --video data/videos/traffic.mp4 \
    --conf 0.5 \
    --line 0.5 \
    --device cuda \
    --save-data

# 参数说明
# --video: 输入视频路径
# --output: 输出视频路径（可选）
# --conf: 检测置信度阈值（默认0.5）
# --line: 检测线位置比例（0-1，默认0.5）
# --device: 运行设备（cuda/cpu）
# --save-data: 保存流量数据到CSV
# --show-progress: 显示进度条（默认开启）
# --no-progress: 关闭进度条
```

### 2. 模型训练

```bash
# 使用默认参数训练
python train.py --data data/traffic_data.csv

# 自定义参数训练
python train.py --data data/traffic_data.csv \
    --epochs 100 \
    --hidden-size 64 \
    --seq-length 10 \
    --device cuda
```

### 3. 启动Web界面

```bash
streamlit run src/visualization/app.py
```

然后在浏览器中打开 `http://localhost:8501`

## 配置说明

配置文件位于 `configs/config.yaml`，主要配置项：

```yaml
# 检测配置
detection:
  model: "yolov8n.pt"      # YOLOv8模型
  conf_threshold: 0.5      # 置信度阈值
  classes: [2, 3, 5, 7]    # 检测类别：car, motorcycle, bus, truck

# 计数配置
counting:
  line_position: 0.5       # 检测线位置
  direction: "both"        # 计数方向

# 预测配置
prediction:
  sequence_length: 10      # 输入序列长度
  hidden_size: 64          # LSTM隐藏层大小
  num_layers: 2            # LSTM层数
```

## 检测类别

系统支持以下车辆类别的检测（基于COCO数据集）：

| 类别ID | 名称 | 说明 |
|--------|------|------|
| 2 | 小汽车 | car |
| 3 | 摩托车 | motorcycle |
| 5 | 公交车 | bus |
| 7 | 卡车 | truck |

## 功能展示

### 视频检测
- 实时检测视频中的车辆
- 在画面上绘制检测框和标签
- 显示检测线位置
- 实时更新计数统计

### 流量分析
- 流量时序图
- 小时流量分布
- 车辆类型分布饼图
- 方向流量对比

### 流量预测
- LSTM模型训练
- 多步流量预测
- 预测结果可视化

## 注意事项

1. **GPU支持**：推荐使用NVIDIA GPU以获得更快的检测速度
2. **视频格式**：支持MP4、AVI、MOV等常见视频格式
3. **内存需求**：处理长视频时可能需要较大内存
4. **模型下载**：首次运行会自动下载YOLOv8模型（约6MB）

## 技术栈

- **目标检测**: YOLOv8 (Ultralytics)
- **深度学习框架**: PyTorch
- **时间序列预测**: LSTM
- **视频处理**: OpenCV
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Plotly
- **Web界面**: Streamlit

## 作者

彭洋 (2205010117)

## 许可证

本项目仅供学习和研究使用。

## 答辩演示准备（推荐）

为避免现场演示时因缺少数据或模型导致中断，先执行一键检查：

```bash
python scripts/check_demo_ready.py
```

检查项包括：
- `data/videos` 中是否有可用演示视频（mp4/avi/mov）
- `data/traffic_data.csv` 是否存在且列完整
- `data/models/best_model.pth` 是否可加载
- `data/models/norm_params.json` 是否存在

若检查失败，可按以下顺序生成演示资产：

```bash
# 1) 从视频生成流量数据
python main.py --video data/videos/<your_video>.mp4 --save-data

# 2) 训练LSTM模型
python train.py --data data/traffic_data.csv

# 3) 启动Web可视化
streamlit run src/visualization/app.py
```
