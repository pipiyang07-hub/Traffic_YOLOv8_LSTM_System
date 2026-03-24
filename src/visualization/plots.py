"""
图表绘制模块
Plotting utilities for visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_traffic_flow(
    data: pd.DataFrame,
    time_column: str = 'timestamp',
    value_column: str = 'count',
    title: str = '交通流量时序图',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制交通流量时序图

    Args:
        data: 数据DataFrame
        time_column: 时间列名
        value_column: 值列名
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column])

    ax.plot(data[time_column], data[value_column], 'b-', linewidth=1.5)
    ax.fill_between(data[time_column], data[value_column], alpha=0.3)

    ax.set_xlabel('时间')
    ax.set_ylabel('车流量')
    ax.set_title(title)

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = '流量预测对比',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制预测结果对比图

    Args:
        actual: 实际值
        predicted: 预测值
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(actual))
    ax.plot(x, actual, 'b-', label='实际值', linewidth=2)
    ax.plot(x, predicted, 'r--', label='预测值', linewidth=2)

    ax.fill_between(x, actual, predicted, alpha=0.2, color='gray')

    ax.set_xlabel('时间步')
    ax.set_ylabel('车流量')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_hourly_distribution(
    data: pd.DataFrame,
    time_column: str = 'timestamp',
    value_column: str = 'count',
    title: str = '小时流量分布',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制小时流量分布图

    Args:
        data: 数据DataFrame
        time_column: 时间列名
        value_column: 值列名
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column])

    # 按小时统计
    data['hour'] = data[time_column].dt.hour
    hourly_data = data.groupby('hour')[value_column].mean()

    # 绘制柱状图
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 24))
    bars = ax.bar(hourly_data.index, hourly_data.values, color=colors, edgecolor='black')

    ax.set_xlabel('小时')
    ax.set_ylabel('平均车流量')
    ax.set_title(title)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_vehicle_type_distribution(
    counts: Dict[str, int],
    title: str = '车辆类型分布',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制车辆类型分布饼图

    Args:
        counts: 车辆类型计数字典
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(counts.keys())
    sizes = list(counts.values())

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = [0.05] * len(labels)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors[:len(labels)],
        autopct='%1.1f%%',
        shadow=True,
        startangle=90
    )
    ax.axis('equal')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_direction_comparison(
    up_counts: Dict[str, int],
    down_counts: Dict[str, int],
    title: str = '方向流量对比',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制方向流量对比图

    Args:
        up_counts: 上行计数
        down_counts: 下行计数
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 获取所有车辆类型
    all_types = sorted(set(list(up_counts.keys()) + list(down_counts.keys())))

    x = np.arange(len(all_types))
    width = 0.35

    up_values = [up_counts.get(t, 0) for t in all_types]
    down_values = [down_counts.get(t, 0) for t in all_types]

    bars1 = ax.bar(x - width/2, up_values, width, label='上行', color='steelblue')
    bars2 = ax.bar(x + width/2, down_values, width, label='下行', color='coral')

    ax.set_xlabel('车辆类型')
    ax.set_ylabel('数量')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = '训练损失曲线',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练历史曲线

    Args:
        history: 训练历史字典
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)

    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r--', label='验证损失', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失 (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
