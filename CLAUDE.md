# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 技术栈

- Python 3.8+
- YOLOv8 (Ultralytics) — 目标检测
- PyTorch — LSTM 时序预测模型
- OpenCV — 视频处理
- Streamlit — Web 可视化界面
- Pandas / NumPy — 数据处理
- scikit-learn — 数据归一化

## 代码规范

- 所有代码必须添加中文注释，包括函数、类、关键逻辑
- 配置统一从 `configs/config.yaml` 读取，通过 `src/utils/data_utils.load_config()` 加载
- 回复语言：中文

## 禁止事项

- 禁止硬编码路径、阈值等参数，应写入 `configs/config.yaml`
- 禁止在代码中写入 API Key、密码等敏感信息
- 禁止新增无关依赖
