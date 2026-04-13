"""
日志工具模块
Logging utilities for the traffic detection system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "traffic_system",
    level: int = logging.INFO,
    log_file: str | None = None 
) -> logging.Logger:
    """
    设置并返回配置好的日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(level)



    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "traffic_system") -> logging.Logger:
    """
    获取已存在的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(name)


# 默认日志记录器
_default_logger = None


def get_default_logger() -> logging.Logger:
    """获取默认日志记录器"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger
