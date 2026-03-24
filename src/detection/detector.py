"""
YOLOv8 目标检测模块
Object detection using YOLOv8
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ultralytics import YOLO

from ..utils.logger import get_default_logger
from ..utils.data_utils import get_vehicle_name

logger = get_default_logger()


class DetectionResult:
    """检测结果类"""

    def __init__(
        self,
        box: Tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str = None
    ):
        """
        初始化检测结果

        Args:
            box: 边界框 (x1, y1, x2, y2)
            confidence: 置信度
            class_id: 类别ID
            class_name: 类别名称
        """
        self.box = box  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name or get_vehicle_name(class_id)

    @property
    def center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """获取边界框底部中心点（用于计数）"""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)

    @property
    def width(self) -> int:
        """边界框宽度"""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        """边界框高度"""
        return self.box[3] - self.box[1]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'box': self.box,
            'center': self.center,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'width': self.width,
            'height': self.height
        }

    def __repr__(self) -> str:
        return f"DetectionResult({self.class_name}, conf={self.confidence:.2f}, box={self.box})"


class YOLOv8Detector:
    """YOLOv8 检测器"""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: List[int] = None
    ):
        """
        初始化YOLOv8检测器

        Args:
            model_path: 模型文件路径
            device: 运行设备 ('cuda' 或 'cpu')
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            classes: 要检测的类别ID列表（None表示所有类别）
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        logger.info(f"正在加载YOLOv8模型: {self.model_path}")

        # 检查模型文件是否存在
        model_path = Path(self.model_path)
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            # 自动下载
            logger.info(f"模型文件不存在，正在下载: {self.model_path}")
            self.model = YOLO(self.model_path)

        # 设置设备
        self.model.to(self.device)

        logger.info(f"模型加载完成，设备: {self.device}")

    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> List[DetectionResult]:
        """
        对单帧图像进行目标检测

        Args:
            image: 输入图像 (BGR格式)
            verbose: 是否输出详细信息

        Returns:
            检测结果列表
        """
        # 执行检测
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=verbose
        )

        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # (x1, y1, x2, y2)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detection = DetectionResult(
                    box=tuple(int(x) for x in box),
                    confidence=conf,
                    class_id=cls_id
                )
                detections.append(detection)

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = False
    ) -> List[List[DetectionResult]]:
        """
        批量检测多帧图像

        Args:
            images: 图像列表
            verbose: 是否输出详细信息

        Returns:
            每帧的检测结果列表
        """
        results = []
        for image in images:
            results.append(self.detect(image, verbose))
        return results

    def get_class_names(self) -> Dict[int, str]:
        """获取模型支持的类别名称"""
        return self.model.names

    def __repr__(self) -> str:
        return f"YOLOv8Detector(model={self.model_path}, device={self.device})"


def draw_detections(
    frame: np.ndarray,
    detections: List[DetectionResult],
    show_conf: bool = True,
    show_labels: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果

    Args:
        frame: 输入图像
        detections: 检测结果列表
        show_conf: 是否显示置信度
        show_labels: 是否显示标签
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    import cv2

    # 不同类别使用不同颜色
    colors = {
        2: (0, 255, 0),      # car - 绿色
        3: (255, 0, 0),      # motorcycle - 蓝色
        5: (0, 255, 255),    # bus - 黄色
        7: (0, 165, 255),    # truck - 橙色
    }

    for det in detections:
        color = colors.get(det.class_id, (0, 255, 0))
        x1, y1, x2, y2 = det.box

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签
        if show_labels:
            label = det.class_name
            if show_conf:
                label += f" {det.confidence:.2f}"

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

        # 绘制中心点
        center = det.center
        cv2.circle(frame, center, 3, color, -1)

    return frame
