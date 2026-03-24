"""
车辆计数模块
Vehicle counting using detection line
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..detection.detector import DetectionResult
from .line import DetectionLine
from ..utils.logger import get_default_logger
from ..utils.data_utils import create_traffic_record, get_vehicle_name

logger = get_default_logger()


class VehicleCounter:
    """车辆计数器"""

    def __init__(
        self,
        detection_line: DetectionLine,
        direction: str = "both"
    ):
        """
        初始化车辆计数器

        Args:
            detection_line: 检测线
            direction: 计数方向 ('up', 'down', 'both')
        """
        self.line = detection_line
        self.direction = direction

        # 计数统计
        self.counts = defaultdict(int)  # 按车型计数
        self.total_counts = defaultdict(int)  # 总计数

        # 方向统计
        self.direction_counts = {
            'up': defaultdict(int),
            'down': defaultdict(int)
        }

        # 记录每个车辆上一帧的位置（使用底部中心点）
        self.prev_positions: Dict[int, Tuple[int, int]] = {}

        # 记录已计数的车辆ID
        self.counted_ids: set = set()

        # 记录列表
        self.records: List[Dict[str, Any]] = []

    def update(
        self,
        detections: List[DetectionResult],
        timestamp: datetime = None
    ) -> Dict[str, int]:
        """
        更新计数（处理一帧检测结果）

        Args:
            detections: 检测结果列表
            timestamp: 时间戳

        Returns:
            当前帧计数结果
        """
        if timestamp is None:
            timestamp = datetime.now()

        frame_counts = defaultdict(int)

        for i, det in enumerate(detections):
            # 使用索引作为简单ID（因为没有跟踪）
            # 使用底部中心点作为判断点
            curr_point = det.bottom_center

            # 检查是否已计数
            if i in self.counted_ids:
                self.prev_positions[i] = curr_point
                continue

            # 获取上一帧位置
            prev_point = self.prev_positions.get(i)

            # 检查是否越过检测线
            crossed = self.line.is_crossed(prev_point, curr_point)

            if crossed:
                # 检查方向是否需要计数
                should_count = (
                    self.direction == "both" or
                    self.direction == crossed
                )

                if should_count:
                    # 计数
                    self.counts[det.class_id] += 1
                    self.total_counts[det.class_id] += 1
                    self.direction_counts[crossed][det.class_id] += 1
                    frame_counts[det.class_id] += 1
                    self.counted_ids.add(i)

                    # 创建记录
                    record = create_traffic_record(
                        timestamp=timestamp,
                        vehicle_type=det.class_name,
                        count=1,
                        direction=crossed,
                        confidence=det.confidence
                    )
                    self.records.append(record)

                    logger.debug(
                        f"计数: {det.class_name} {crossed} "
                        f"(总计: {self.total_counts[det.class_id]})"
                    )

            # 更新位置记录
            self.prev_positions[i] = curr_point

        return dict(frame_counts)

    def update_with_tracking(
        self,
        detections: List[DetectionResult],
        track_ids: List[int],
        timestamp: datetime = None
    ) -> Dict[str, int]:
        """
        使用跟踪ID更新计数

        Args:
            detections: 检测结果列表
            track_ids: 跟踪ID列表
            timestamp: 时间戳

        Returns:
            当前帧计数结果
        """
        if timestamp is None:
            timestamp = datetime.now()

        frame_counts = defaultdict(int)

        for det, track_id in zip(detections, track_ids):
            curr_point = det.bottom_center

            # 检查是否已计数
            if track_id in self.counted_ids:
                self.prev_positions[track_id] = curr_point
                continue

            # 获取上一帧位置
            prev_point = self.prev_positions.get(track_id)

            # 检查是否越过检测线
            crossed = self.line.is_crossed(prev_point, curr_point)

            if crossed:
                should_count = (
                    self.direction == "both" or
                    self.direction == crossed
                )

                if should_count:
                    self.counts[det.class_id] += 1
                    self.total_counts[det.class_id] += 1
                    self.direction_counts[crossed][det.class_id] += 1
                    frame_counts[det.class_id] += 1
                    self.counted_ids.add(track_id)

                    record = create_traffic_record(
                        timestamp=timestamp,
                        vehicle_type=det.class_name,
                        count=1,
                        direction=crossed,
                        confidence=det.confidence
                    )
                    self.records.append(record)

            # 更新位置记录
            self.prev_positions[track_id] = curr_point

        return dict(frame_counts)

    def reset_frame(self):
        """重置帧计数（保留总计数和记录）"""
        self.counts.clear()
        self.prev_positions.clear()
        self.counted_ids.clear()

    def reset_all(self):
        """重置所有计数"""
        self.counts.clear()
        self.total_counts.clear()
        self.direction_counts = {
            'up': defaultdict(int),
            'down': defaultdict(int)
        }
        self.prev_positions.clear()
        self.counted_ids.clear()
        self.records.clear()

    def get_total_count(self) -> int:
        """获取总计数"""
        return sum(self.total_counts.values())

    def get_counts_by_type(self) -> Dict[str, int]:
        """按车型获取计数"""
        return {
            get_vehicle_name(class_id): count
            for class_id, count in self.total_counts.items()
        }

    def get_counts_by_direction(self) -> Dict[str, Dict[str, int]]:
        """按方向获取计数"""
        return {
            'up': {
                get_vehicle_name(class_id): count
                for class_id, count in self.direction_counts['up'].items()
            },
            'down': {
                get_vehicle_name(class_id): count
                for class_id, count in self.direction_counts['down'].items()
            }
        }

    def get_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total': self.get_total_count(),
            'by_type': self.get_counts_by_type(),
            'by_direction': self.get_counts_by_direction(),
            'record_count': len(self.records)
        }

    def __repr__(self) -> str:
        return f"VehicleCounter(total={self.get_total_count()}, line={self.line.name})"


class CountingVisualizer:
    """计数可视化工具"""

    def __init__(self, counter: VehicleCounter):
        self.counter = counter

    def draw_counting_info(
        self,
        frame: np.ndarray,
        show_line: bool = True
    ) -> np.ndarray:
        """
        在帧上绘制计数信息

        Args:
            frame: 输入帧
            show_line: 是否显示检测线

        Returns:
            绘制后的帧
        """
        import cv2

        # 绘制检测线
        if show_line:
            start, end = self.counter.line.get_draw_coords()
            cv2.line(frame, start, end, (0, 0, 255), 2)

        # 绘制计数信息
        y_offset = 30
        total = self.counter.get_total_count()

        # 总计数
        cv2.putText(
            frame,
            f"Total: {total}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_offset += 30

        # 按车型计数
        for vehicle_type, count in self.counter.get_counts_by_type().items():
            cv2.putText(
                frame,
                f"{vehicle_type}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
            y_offset += 25

        return frame
