"""
数据处理工具模块
Data processing utilities
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .logger import get_default_logger

logger = get_default_logger()


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"已加载配置文件: {config_path}")
    return config


def save_to_csv(
    data: List[Dict[str, Any]],
    output_path: str,
    mode: str = 'w'
) -> str:
    """
    将数据保存到CSV文件

    Args:
        data: 数据列表，每个元素是一个字典
        output_path: 输出文件路径
        mode: 写入模式，'w'覆盖，'a'追加

    Returns:
        保存的文件路径
    """
    if not data:
        logger.warning("数据为空，跳过保存")
        return output_path

    path = Path(output_path)
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    # 如果是追加模式且文件存在，不写header
    header = True
    if mode == 'a' and path.exists():
        header = False

    df.to_csv(path, mode=mode, index=False, header=header, encoding='utf-8') # type: ignore
    logger.info(f"数据已保存: {path} ({len(data)} 条记录)")

    return str(path)


def load_traffic_data(data_path: str) -> pd.DataFrame:
    """
    加载交通流量数据

    Args:
        data_path: 数据文件路径

    Returns:
        DataFrame
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(path)
    logger.info(f"已加载数据: {path} ({len(df)} 条记录)")
    return df


def create_traffic_record(
    timestamp: datetime,
    vehicle_type: str,
    count: int,
    direction: str = "unknown",
    confidence: float = 0.0
) -> Dict[str, Any]:
    """
    创建单条交通记录

    Args:
        timestamp: 时间戳
        vehicle_type: 车辆类型
        count: 数量
        direction: 方向
        confidence: 置信度

    Returns:
        记录字典
    """
    return {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'vehicle_type': vehicle_type,
        'count': count,
        'direction': direction,
        'confidence': round(confidence, 4)
    }


def aggregate_by_time(
    df: pd.DataFrame,
    time_interval: str = '1min'
) -> pd.DataFrame:
    """
    按时间间隔聚合流量数据

    Args:
        df: 原始数据
        time_interval: 时间间隔（如 '1min', '5min', '1h'）

    Returns:
        聚合后的DataFrame
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    aggregated = df.groupby([
        pd.Grouper(freq=time_interval),
        'vehicle_type'
    ])['count'].sum().reset_index()

    return aggregated


def normalize_data(
    data: pd.Series,
    method: str = 'minmax'
) -> tuple:
    """
    数据归一化

    Args:
        data: 输入数据
        method: 归一化方法 ('minmax' 或 'standard')

    Returns:
        (归一化后的数据, 归一化参数)
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    elif method == 'standard':
        mean_val = data.mean()
        std_val = data.std()
        normalized = (data - mean_val) / (std_val + 1e-8)
        params = {'mean': mean_val, 'std': std_val}
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

    return normalized, params


def denormalize_data(
    normalized_data: pd.Series,
    params: Dict[str, float],
    method: str = 'minmax'
) -> pd.Series:
    """
    数据反归一化

    Args:
        normalized_data: 归一化后的数据
        params: 归一化参数
        method: 归一化方法

    Returns:
        原始尺度的数据
    """
    if method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    elif method == 'standard':
        return normalized_data * params['std'] + params['mean']
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


# COCO数据集类别映射
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    # ... 更多类别
}

# 车辆类别映射（中文）
VEHICLE_NAMES = {
    2: '小汽车',
    3: '摩托车',
    5: '公交车',
    7: '卡车'
}


def get_vehicle_name(class_id: int) -> str:
    """
    获取车辆类型名称

    Args:
        class_id: COCO类别ID

    Returns:
        车辆名称
    """
    return VEHICLE_NAMES.get(class_id, COCO_CLASSES.get(class_id, 'unknown'))
