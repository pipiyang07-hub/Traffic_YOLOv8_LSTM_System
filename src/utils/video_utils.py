"""
视频处理工具模块
Video processing utilities
"""

import cv2
from pathlib import Path
from typing import Optional, Generator, Tuple
import numpy as np

from .logger import get_default_logger

logger = get_default_logger()


class VideoReader:
    """视频读取器"""

    def __init__(self, video_path: str):
        """
        初始化视频读取器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = Path(video_path)
        self.cap = None
        self._open()

    def _open(self):
        """打开视频文件"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_path}")

        logger.info(f"已打开视频: {self.video_path}")

    @property
    def width(self) -> int:
        """视频宽度"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """视频高度"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """视频帧率"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """总帧数"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        """视频时长（秒）"""
        return self.frame_count / self.fps if self.fps > 0 else 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取下一帧

        Returns:
            (success, frame) 元组
        """
        return self.cap.read()

    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """
        生成器方式读取所有帧

        Yields:
            视频帧
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def seek(self, frame_idx: int):
        """
        跳转到指定帧

        Args:
            frame_idx: 帧索引
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
            logger.info("视频读取器已释放")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


class VideoWriter:
    """视频写入器"""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        fourcc: str = "mp4v"
    ):
        """
        初始化视频写入器

        Args:
            output_path: 输出文件路径
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            fourcc: 编码格式
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.writer = None
        self._open()

    def _open(self):
        """创建视频写入器"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {self.output_path}")

        logger.info(f"已创建视频写入器: {self.output_path}")

    def write_frame(self, frame: np.ndarray):
        """
        写入一帧

        Args:
            frame: 视频帧
        """
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def write_frames(self, frames: list):
        """
        写入多帧

        Args:
            frames: 视频帧列表
        """
        for frame in frames:
            self.write_frame(frame)

    def release(self):
        """释放资源"""
        if self.writer:
            self.writer.release()
            logger.info(f"视频已保存: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


def resize_frame(frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    缩放视频帧

    Args:
        frame: 输入帧
        scale: 缩放比例

    Returns:
        缩放后的帧
    """
    if scale == 1.0:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))


def draw_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在帧上绘制边界框

    Args:
        frame: 输入帧
        box: (x1, y1, x2, y2) 边界框坐标
        label: 标签文字
        color: 颜色 (B, G, R)
        thickness: 线宽

    Returns:
        绘制后的帧
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    return frame


def draw_line(
    frame: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    在帧上绘制检测线

    Args:
        frame: 输入帧
        start: 起点 (x, y)
        end: 终点 (x, y)
        color: 颜色 (B, G, R)
        thickness: 线宽

    Returns:
        绘制后的帧
    """
    cv2.line(frame, start, end, color, thickness)
    return frame
