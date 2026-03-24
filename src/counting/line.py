"""
检测线模块
Detection line for vehicle counting
"""

import numpy as np
from typing import Tuple, Optional


class DetectionLine:
    """检测线类"""

    def __init__(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        name: str = "line"
    ):
        """
        初始化检测线

        Args:
            start: 起点 (x, y)
            end: 终点 (x, y)
            name: 检测线名称
        """
        self.start = start
        self.end = end
        self.name = name

        # 计算线段向量
        self._vector = np.array([
            end[0] - start[0],
            end[1] - start[1]
        ], dtype=np.float64)

        # 线段长度
        self._length = np.linalg.norm(self._vector)

        # 单位向量
        if self._length > 0:
            self._unit_vector = self._vector / self._length
        else:
            self._unit_vector = np.array([0, 1])

    @classmethod
    def from_position_ratio(
        cls,
        frame_width: int,
        frame_height: int,
        position_ratio: float = 0.5,
        name: str = "line"
    ) -> "DetectionLine":
        """
        根据位置比例创建水平检测线

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            position_ratio: 位置比例 (0-1)，0为顶部，1为底部
            name: 检测线名称

        Returns:
            检测线实例
        """
        y = int(frame_height * position_ratio)
        start = (0, y)
        end = (frame_width, y)
        return cls(start, end, name)

    def distance_to_point(self, point: Tuple[int, int]) -> float:
        """
        计算点到检测线的垂直距离

        Args:
            point: 点坐标 (x, y)

        Returns:
            距离（带符号，正负表示在线的两侧）
        """
        # 向量从起点到点
        point_vec = np.array([
            point[0] - self.start[0],
            point[1] - self.start[1]
        ])

        # 使用叉积计算有符号距离
        cross = point_vec[0] * self._unit_vector[1] - point_vec[1] * self._unit_vector[0]
        return cross

    def is_crossed(
        self,
        prev_point: Optional[Tuple[int, int]],
        curr_point: Tuple[int, int]
    ) -> Optional[str]:
        """
        判断是否越过检测线

        Args:
            prev_point: 上一帧的点位置
            curr_point: 当前帧的点位置

        Returns:
            越过方向 ('up', 'down', None)
        """
        if prev_point is None:
            return None

        prev_dist = self.distance_to_point(prev_point)
        curr_dist = self.distance_to_point(curr_point)

        # 判断是否越过（符号变化）
        if prev_dist * curr_dist < 0:
            # 从正到负表示向上，从负到正表示向下
            if prev_dist > 0:
                return 'up'
            else:
                return 'down'

        return None

    def get_draw_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """获取绘制坐标"""
        return self.start, self.end

    def __repr__(self) -> str:
        return f"DetectionLine({self.name}, {self.start} -> {self.end})"


class MultiLineManager:
    """多检测线管理器"""

    def __init__(self):
        self.lines = {}
        self.cross_records = {}  # 记录每个目标是否已跨过各条线

    def add_line(self, line: DetectionLine):
        """添加检测线"""
        self.lines[line.name] = line
        self.cross_records[line.name] = {}

    def create_horizontal_lines(
        self,
        frame_width: int,
        frame_height: int,
        positions: list = [0.3, 0.5, 0.7]
    ):
        """
        创建多条水平检测线

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            positions: 位置比例列表
        """
        for i, pos in enumerate(positions):
            line = DetectionLine.from_position_ratio(
                frame_width, frame_height, pos, f"line_{i}"
            )
            self.add_line(line)

    def check_cross(
        self,
        line_name: str,
        object_id: int,
        prev_point: Optional[Tuple[int, int]],
        curr_point: Tuple[int, int]
    ) -> Optional[str]:
        """
        检查目标是否越过指定检测线

        Args:
            line_name: 检测线名称
            object_id: 目标ID
            prev_point: 上一帧位置
            curr_point: 当前帧位置

        Returns:
            越过方向或None
        """
        if line_name not in self.lines:
            return None

        # 检查是否已跨过
        if object_id in self.cross_records[line_name]:
            return None

        line = self.lines[line_name]
        direction = line.is_crossed(prev_point, curr_point)

        if direction:
            # 记录已跨过
            self.cross_records[line_name][object_id] = direction

        return direction

    def reset_records(self):
        """重置所有跨线记录"""
        for line_name in self.cross_records:
            self.cross_records[line_name] = {}

    def get_line(self, line_name: str) -> Optional[DetectionLine]:
        """获取指定检测线"""
        return self.lines.get(line_name)

    def get_all_lines(self) -> list:
        """获取所有检测线"""
        return list(self.lines.values())
