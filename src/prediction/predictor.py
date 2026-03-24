"""
流量预测模块
Traffic flow prediction
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .lstm_model import LSTMModel
from .trainer import LSTMTrainer
from ..utils.logger import get_default_logger
from ..utils.data_utils import load_traffic_data, normalize_data, denormalize_data

logger = get_default_logger()


class TrafficPredictor:
    """交通流量预测器"""

    def __init__(
        self,
        model: Optional[LSTMModel] = None,
        model_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        初始化预测器

        Args:
            model: 已训练的模型
            model_path: 模型文件路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.norm_params = None
        self.norm_method = 'minmax'

        # 如果提供了模型路径，加载模型
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载模型

        Args:
            model_path: 模型文件路径
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(path, map_location=self.device)

        # 从checkpoint获取模型配置
        config = checkpoint.get('model_config', {})
        self.model = LSTMModel(
            input_size=config.get('input_size', 1),
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 2),
            output_size=config.get('output_size', 1),
            bidirectional=config.get('bidirectional', False)
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"模型已加载: {model_path}")

    def set_normalization_params(
        self,
        min_val: float,
        max_val: float,
        method: str = 'minmax'
    ):
        """
        设置归一化参数

        Args:
            min_val: 最小值
            max_val: 最大值
            method: 归一化方法
        """
        self.norm_params = {'min': min_val, 'max': max_val}
        self.norm_method = method

    def prepare_input(
        self,
        data: np.ndarray,
        seq_length: int
    ) -> torch.Tensor:
        """
        准备模型输入

        Args:
            data: 原始数据
            seq_length: 序列长度

        Returns:
            模型输入张量
        """
        # 取最后 seq_length 个数据点
        if len(data) < seq_length:
            raise ValueError(f"数据长度 {len(data)} 小于序列长度 {seq_length}")

        x = data[-seq_length:]

        # 归一化
        if self.norm_params:
            x = (x - self.norm_params['min']) / (
                self.norm_params['max'] - self.norm_params['min'] + 1e-8
            )

        # 转换为张量
        x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        x = x.to(self.device)

        return x

    def predict(
        self,
        data: np.ndarray,
        seq_length: int = 10
    ) -> float:
        """
        单步预测

        Args:
            data: 历史数据
            seq_length: 序列长度

        Returns:
            预测值
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        self.model.eval()

        with torch.no_grad():
            x = self.prepare_input(data, seq_length)
            output, _ = self.model(x)
            prediction = output.cpu().numpy().flatten()[0]

        # 反归一化
        if self.norm_params:
            prediction = prediction * (
                self.norm_params['max'] - self.norm_params['min']
            ) + self.norm_params['min']

        return prediction

    def predict_multi_step(
        self,
        data: np.ndarray,
        steps: int = 10,
        seq_length: int = 10
    ) -> np.ndarray:
        """
        多步预测

        Args:
            data: 历史数据
            steps: 预测步数
            seq_length: 序列长度

        Returns:
            预测结果数组
        """
        predictions = []
        current_data = data.copy()

        for _ in range(steps):
            pred = self.predict(current_data, seq_length)
            predictions.append(pred)
            # 将预测值加入数据序列
            current_data = np.append(current_data[1:], pred)

        return np.array(predictions)

    def predict_with_confidence(
        self,
        data: np.ndarray,
        seq_length: int = 10,
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        带置信区间的预测（使用MC Dropout）

        Args:
            data: 历史数据
            seq_length: 序列长度
            n_samples: 采样次数

        Returns:
            (预测值, 标准差)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 启用Dropout
        self.model.train()

        predictions = []
        with torch.no_grad():
            x = self.prepare_input(data, seq_length)

            for _ in range(n_samples):
                output, _ = self.model(x)
                pred = output.cpu().numpy().flatten()[0]

                if self.norm_params:
                    pred = pred * (
                        self.norm_params['max'] - self.norm_params['min']
                    ) + self.norm_params['min']

                predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = predictions.mean()
        std_pred = predictions.std()

        return mean_pred, std_pred


class TrafficDataProcessor:
    """交通数据处理工具"""

    @staticmethod
    def load_and_process(
        data_path: str,
        time_column: str = 'timestamp',
        value_column: str = 'count',
        aggregation: str = '5min'
    ) -> pd.DataFrame:
        """
        加载并处理交通数据

        Args:
            data_path: 数据文件路径
            time_column: 时间列名
            value_column: 值列名
            aggregation: 聚合时间间隔

        Returns:
            处理后的DataFrame
        """
        # 加载数据
        df = load_traffic_data(data_path)

        # 转换时间列
        df[time_column] = pd.to_datetime(df[time_column])

        # 按时间聚合
        df = df.set_index(time_column)
        aggregated = df.resample(aggregation)[value_column].sum().reset_index()

        return aggregated

    @staticmethod
    def create_features(df: pd.DataFrame, time_column: str = 'timestamp') -> pd.DataFrame:
        """
        创建时间特征

        Args:
            df: DataFrame
            time_column: 时间列名

        Returns:
            添加特征后的DataFrame
        """
        df = df.copy()
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    @staticmethod
    def get_sequence_data(
        df: pd.DataFrame,
        value_column: str = 'count',
        seq_length: int = 10
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        获取序列数据和归一化参数

        Args:
            df: DataFrame
            value_column: 值列名
            seq_length: 序列长度

        Returns:
            (数据数组, 归一化参数)
        """
        data = df[value_column].values.astype(np.float32)
        normalized, params = normalize_data(pd.Series(data), method='minmax')
        return normalized.values, params


def predict_future_traffic(
    data_path: str,
    model_path: str,
    predict_steps: int = 10,
    seq_length: int = 10
) -> List[Dict[str, Any]]:
    """
    预测未来交通流量的便捷函数

    Args:
        data_path: 数据文件路径
        model_path: 模型文件路径
        predict_steps: 预测步数
        seq_length: 序列长度

    Returns:
        预测结果列表
    """
    # 加载数据
    processor = TrafficDataProcessor()
    df = processor.load_and_process(data_path)

    # 获取序列数据
    data, norm_params = processor.get_sequence_data(df, seq_length=seq_length)

    # 创建预测器
    predictor = TrafficPredictor(model_path=model_path)
    predictor.set_normalization_params(
        norm_params['min'], norm_params['max']
    )

    # 预测
    predictions = predictor.predict_multi_step(
        data, steps=predict_steps, seq_length=seq_length
    )

    # 格式化结果
    last_time = df['timestamp'].iloc[-1]
    results = []
    for i, pred in enumerate(predictions):
        future_time = last_time + timedelta(minutes=5 * (i + 1))
        results.append({
            'timestamp': future_time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_flow': int(round(pred))
        })

    return results
