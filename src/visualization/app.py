"""
Streamlit 可视化应用
Web visualization interface using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import os

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.detector import YOLOv8Detector, draw_detections
from src.counting.line import DetectionLine
from src.counting.counter import VehicleCounter, CountingVisualizer
from src.prediction.lstm_model import LSTMModel, prepare_data
from src.prediction.predictor import TrafficPredictor
from src.utils.video_utils import VideoReader, VideoWriter, draw_line
from src.utils.data_utils import load_config, save_to_csv
from src.visualization.plots import (
    plot_traffic_flow,
    plot_vehicle_type_distribution,
    plot_direction_comparison,
    plot_hourly_distribution,
    plot_prediction
)

# 页面配置
st.set_page_config(
    page_title="交通流量检测与预测系统",
    page_icon="🚗",
    layout="wide"
)


@st.cache_resource
def load_detector(config):
    """加载检测模型（缓存）"""
    return YOLOv8Detector(
        model_path=config['detection']['model'],
        device=config['detection']['device'],
        conf_threshold=config['detection']['conf_threshold'],
        classes=config['detection']['classes']
    )


@st.cache_resource
def load_predictor(model_path):
    """加载预测模型（缓存）"""
    if Path(model_path).exists():
        return TrafficPredictor(model_path=model_path)
    return None


def main():
    """主函数"""
    st.title("🚗 基于YOLOv8与LSTM的交通目标检测与流量预测系统")

    # 侧边栏
    st.sidebar.title("功能选择")
    page = st.sidebar.radio(
        "选择功能",
        ["视频检测与计数", "流量数据分析", "流量预测", "系统设置"]
    )

    # 加载配置
    try:
        config = load_config("configs/config.yaml")
    except FileNotFoundError:
        st.error("配置文件不存在，请检查 configs/config.yaml")
        return

    if page == "视频检测与计数":
        show_detection_page(config)
    elif page == "流量数据分析":
        show_analysis_page(config)
    elif page == "流量预测":
        show_prediction_page(config)
    else:
        show_settings_page(config)


def show_detection_page(config):
    """视频检测与计数页面"""
    st.header("📹 视频检测与计数")

    col1, col2 = st.columns([2, 1])

    with col1:
        # 视频输入
        st.subheader("输入视频")
        video_source = st.radio(
            "选择视频来源",
            ["上传视频文件", "选择示例视频"],
            horizontal=True
        )

        video_path = None
        if video_source == "上传视频文件":
            uploaded_file = st.file_uploader(
                "上传交通视频",
                type=['mp4', 'avi', 'mov'],
                help="支持 MP4, AVI, MOV 格式"
            )
            if uploaded_file:
                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
        else:
            # 列出示例视频
            video_dir = Path(config['paths']['video_input'])
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
                if videos:
                    selected_video = st.selectbox(
                        "选择示例视频",
                        options=[v.name for v in videos]
                    )
                    video_path = str(video_dir / selected_video)
                else:
                    st.info("示例视频目录为空，请上传视频或添加示例视频到 data/videos/ 目录")

        # 检测参数
        st.subheader("检测参数")
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=float(config['detection']['conf_threshold']),
            step=0.05
        )

        line_position = st.slider(
            "检测线位置",
            min_value=0.1,
            max_value=0.9,
            value=float(config['counting']['line_position']),
            step=0.05,
            help="检测线在视频中的垂直位置比例"
        )

    with col2:
        # 开始检测按钮
        start_detection = st.button("开始检测", type="primary")

        if video_path and start_detection:
            st.subheader("检测结果")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 加载模型
            with st.spinner("加载模型..."):
                detector = load_detector(config)

            # 打开视频
            video_reader = VideoReader(video_path)
            line = DetectionLine.from_position_ratio(
                video_reader.width,
                video_reader.height,
                line_position
            )
            counter = VehicleCounter(line)

            # 显示检测线预览
            ret, preview_frame = video_reader.read_frame()
            if ret:
                preview_frame = draw_line(
                    preview_frame,
                    line.start,
                    line.end,
                    color=(0, 0, 255),
                    thickness=2
                )
                st.image(preview_frame, caption="检测线预览", channels="BGR")
                video_reader.seek(0)

            # 处理视频
            total_frames = video_reader.frame_count
            processed_frames = 0

            # 输出视频
            output_dir = Path(config['paths']['output'])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

            video_writer = VideoWriter(
                str(output_path),
                video_reader.width,
                video_reader.height,
                video_reader.fps
            )

            while True:
                ret, frame = video_reader.read_frame()
                if not ret:
                    break

                # 检测
                detections = detector.detect(frame)

                # 计数
                counter.update(detections)

                # 绘制结果
                frame = draw_detections(frame, detections)
                frame = draw_line(frame, line.start, line.end, color=(0, 0, 255), thickness=2)

                # 绘制计数信息
                visualizer = CountingVisualizer(counter)
                frame = visualizer.draw_counting_info(frame, show_line=False)

                video_writer.write_frame(frame)

                processed_frames += 1
                progress = processed_frames / total_frames
                progress_bar.progress(progress)
                status_text.text(f"处理进度: {processed_frames}/{total_frames} 帧")

            video_reader.release()
            video_writer.release()

            # 显示结果
            st.success(f"检测完成！共处理 {processed_frames} 帧")

            # 显示统计
            st.subheader("统计结果")
            summary = counter.get_summary()

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("总车流量", summary['total'])
            with col_b:
                st.metric("车型数", len(summary['by_type']))
            with col_c:
                st.metric("记录数", summary['record_count'])

            # 车辆类型分布
            if summary['by_type']:
                fig = plot_vehicle_type_distribution(summary['by_type'])
                st.pyplot(fig)

            # 保存数据
            if st.button("保存检测数据"):
                save_path = Path(config['paths']['traffic_data'])
                save_to_csv(counter.get_records(), str(save_path), mode='a')
                st.success(f"数据已保存到 {save_path}")

            # 下载结果视频
            with open(output_path, 'rb') as f:
                st.download_button(
                    "下载检测结果视频",
                    f,
                    file_name=output_path.name,
                    mime='video/mp4'
                )


def show_analysis_page(config):
    """流量数据分析页面"""
    st.header("📊 流量数据分析")

    data_path = Path(config['paths']['traffic_data'])
    if not data_path.exists():
        st.warning("暂无流量数据，请先进行视频检测")
        return

    # 加载数据
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    st.subheader("数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总记录数", len(df))
    with col2:
        st.metric("总车流量", df['count'].sum())
    with col3:
        st.metric("时间跨度", f"{df['timestamp'].max() - df['timestamp'].min()}")
    with col4:
        st.metric("平均流量", f"{df['count'].mean():.1f}")

    # 流量时序图
    st.subheader("流量时序图")
    time_df = df.groupby('timestamp')['count'].sum().reset_index()
    fig = plot_traffic_flow(time_df)
    st.pyplot(fig)

    # 小时分布
    st.subheader("小时流量分布")
    df['hour'] = df['timestamp'].dt.hour
    fig = plot_hourly_distribution(df)
    st.pyplot(fig)

    # 车辆类型分布
    st.subheader("车辆类型分布")
    type_counts = df.groupby('vehicle_type')['count'].sum().to_dict()
    fig = plot_vehicle_type_distribution(type_counts)
    st.pyplot(fig)

    # 数据表格
    st.subheader("数据明细")
    st.dataframe(df, use_container_width=True)


def show_prediction_page(config):
    """流量预测页面"""
    st.header("🔮 流量预测")

    data_path = Path(config['paths']['traffic_data'])
    model_path = Path(config['paths']['models']) / "best_model.pth"

    if not data_path.exists():
        st.warning("暂无流量数据，请先进行视频检测")
        return

    # 加载数据
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 参数设置
    st.subheader("预测参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        seq_length = st.number_input("序列长度", min_value=5, max_value=50, value=10)
    with col2:
        predict_steps = st.number_input("预测步数", min_value=1, max_value=30, value=10)
    with col3:
        hidden_size = st.number_input("隐藏层大小", min_value=16, max_value=256, value=64)

    # 准备数据
    time_df = df.groupby('timestamp')['count'].sum().reset_index()
    data = time_df['count'].values.astype(np.float32)

    # 归一化参数
    data_min, data_max = data.min(), data.max()
    normalized_data = (data - data_min) / (data_max - data_min + 1e-8)

    # 检查是否有训练好的模型
    if model_path.exists():
        st.success("发现已训练模型")
        use_existing = st.checkbox("使用现有模型", value=True)
    else:
        use_existing = False

    if use_existing:
        # 加载模型进行预测
        predictor = load_predictor(str(model_path))
        predictor.set_normalization_params(data_min, data_max)

        # 预测
        predictions = predictor.predict_multi_step(
            normalized_data,
            steps=predict_steps,
            seq_length=seq_length
        )

        # 显示预测结果
        st.subheader("预测结果")
        fig = plot_prediction(data[-predict_steps:], predictions, title="流量预测")
        st.pyplot(fig)

        # 预测数据表
        st.subheader("预测数据")
        last_time = time_df['timestamp'].iloc[-1]
        pred_df = pd.DataFrame({
            '时间': [last_time + pd.Timedelta(minutes=5*(i+1)) for i in range(predict_steps)],
            '预测车流量': predictions.round().astype(int)
        })
        st.dataframe(pred_df)

    else:
        # 训练新模型
        if st.button("训练模型"):
            from src.prediction.trainer import LSTMTrainer

            with st.spinner("训练模型中..."):
                # 准备数据
                X_train, y_train, X_test, y_test = prepare_data(
                    normalized_data, seq_length
                )

                # 创建模型
                model = LSTMModel(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=2,
                    output_size=1
                )

                # 训练
                trainer = LSTMTrainer(
                    model,
                    learning_rate=0.001,
                    device='cpu'
                )

                history = trainer.fit(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=50,
                    batch_size=16,
                    verbose=False
                )

                # 保存模型
                trainer.save_model("best_model.pth")

                # 显示训练历史
                st.subheader("训练损失曲线")
                fig, ax = plt.subplots()
                ax.plot(history['train_loss'], label='训练损失')
                ax.plot(history['val_loss'], label='验证损失')
                ax.legend()
                st.pyplot(fig)

                st.success("模型训练完成！")


def show_settings_page(config):
    """系统设置页面"""
    st.header("⚙️ 系统设置")

    st.subheader("当前配置")
    st.json(config)

    st.subheader("路径说明")
    st.markdown(f"""
    - **视频输入目录**: `{config['paths']['video_input']}`
    - **输出目录**: `{config['paths']['output']}`
    - **流量数据**: `{config['paths']['traffic_data']}`
    - **模型目录**: `{config['paths']['models']}`
    """)

    st.subheader("检测类别")
    class_names = {2: '小汽车', 3: '摩托车', 5: '公交车', 7: '卡车'}
    for class_id in config['detection']['classes']:
        st.markdown(f"- {class_id}: {class_names.get(class_id, '未知')}")


if __name__ == "__main__":
    main()
