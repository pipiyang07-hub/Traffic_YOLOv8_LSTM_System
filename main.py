"""
交通目标检测与流量统计主程序
Main entry point for traffic detection and counting

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --output path/to/output.mp4
    python main.py --video path/to/video.mp4 --conf 0.5 --line 0.5
"""

import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.detection.detector import YOLOv8Detector, draw_detections
from src.counting.line import DetectionLine
from src.counting.counter import VehicleCounter, CountingVisualizer
from src.utils.video_utils import VideoReader, VideoWriter, draw_line
from src.utils.data_utils import load_config, save_to_csv
from src.utils.logger import setup_logger

logger = setup_logger(log_file="logs/detection.log")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='交通目标检测与流量统计系统'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='输入视频文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出视频文件路径'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='检测置信度阈值'
    )
    parser.add_argument(
        '--line',
        type=float,
        default=None,
        help='检测线位置 (0-1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='运行设备'
    )
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='是否保存流量数据到CSV'
    )
    parser.add_argument(
        '--show-progress',
        dest='show_progress',
        action='store_true',
        help='显示进度条'
    )
    parser.add_argument(
        '--no-progress',
        dest='show_progress',
        action='store_false',
        help='关闭进度条'
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 覆盖命令行参数
    if args.conf is not None:
        config['detection']['conf_threshold'] = args.conf
    if args.line is not None:
        config['counting']['line_position'] = args.line
    if args.device is not None:
        config['detection']['device'] = args.device

    # 检查视频文件
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return

    logger.info(f"处理视频: {video_path}")

    # 初始化检测器
    logger.info("初始化YOLOv8检测器...")
    detector = YOLOv8Detector(
        model_path=config['detection']['model'],
        device=config['detection']['device'],
        conf_threshold=config['detection']['conf_threshold'],
        iou_threshold=config['detection']['iou_threshold'],
        classes=config['detection']['classes']
    )

    # 打开视频
    video_reader = VideoReader(str(video_path))
    logger.info(
        f"视频信息: {video_reader.width}x{video_reader.height}, "
        f"{video_reader.fps:.1f}fps, {video_reader.frame_count}帧"
    )

    # 创建检测线
    line = DetectionLine.from_position_ratio(
        video_reader.width,
        video_reader.height,
        config['counting']['line_position']
    )
    logger.info(f"检测线位置: {line}")

    # 创建计数器
    counter = VehicleCounter(line, direction=config['counting']['direction'])

    # 输出视频路径
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(config['paths']['output'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    # 创建视频写入器
    video_writer = VideoWriter(
        str(output_path),
        video_reader.width,
        video_reader.height,
        video_reader.fps
    )

    # 创建可视化工具
    visualizer = CountingVisualizer(counter)

    # 处理视频
    logger.info("开始处理视频...")
    total_frames = video_reader.frame_count

    iterator = range(total_frames)
    if args.show_progress:
        iterator = tqdm(iterator, desc="处理中")

    for _ in iterator:
        ret, frame = video_reader.read_frame()
        if not ret:
            break

        # 检测 + 跟踪（优先使用跟踪ID计数）
        detections, track_ids = detector.detect_with_tracking(frame)

        # 计数（有跟踪ID则走跟踪计数，失败时自动回退）
        if track_ids is not None and len(track_ids) == len(detections):
            counter.update_with_tracking(detections, track_ids)
        else:
            counter.update(detections)

        # 绘制检测结果
        frame = draw_detections(
            frame,
            detections,
            show_conf=config['visualization']['show_conf'],
            show_labels=config['visualization']['show_labels'],
            thickness=config['visualization']['line_thickness']
        )

        # 绘制检测线
        frame = draw_line(
            frame,
            line.start,
            line.end,
            color=(0, 0, 255),
            thickness=2
        )

        # 绘制计数信息
        frame = visualizer.draw_counting_info(frame, show_line=False)

        # 写入视频
        video_writer.write_frame(frame)

    # 释放资源
    video_reader.release()
    video_writer.release()

    # 输出统计结果
    summary = counter.get_summary()
    logger.info("=" * 50)
    logger.info("检测完成！统计结果:")
    logger.info(f"  总车流量: {summary['total']}")
    logger.info(f"  各类型: {summary['by_type']}")
    logger.info(f"  按方向: {summary['by_direction']}")
    logger.info("=" * 50)

    # 保存数据
    if args.save_data:
        data_path = Path(config['paths']['traffic_data'])
        save_to_csv(counter.get_records(), str(data_path), mode='a')
        logger.info(f"数据已保存到: {data_path}")

    print(f"\n检测完成！")
    print(f"输出视频: {output_path}")
    print(f"总车流量: {summary['total']}")
    print(f"车型分布: {summary['by_type']}")


if __name__ == "__main__":
    main()
