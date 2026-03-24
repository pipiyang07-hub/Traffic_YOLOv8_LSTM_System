"""
LSTM 流量预测模型
Traffic flow prediction using LSTM
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """LSTM时间序列预测模型"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        初始化LSTM模型

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h_0, c_0)

        Returns:
            (output, hidden) 输出和新的隐藏状态
        """
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态

        Args:
            batch_size: 批次大小
            device: 设备

        Returns:
            (h_0, c_0) 初始隐藏状态
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        return (h0, c0)


class BiLSTMModel(LSTMModel):
    """双向LSTM模型"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=True
        )


class StackedLSTMModel(nn.Module):
    """堆叠LSTM模型（带残差连接）"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 3,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 第一层
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 中间层（带残差）
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_layers - 1)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层LSTM
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out = self.layer_norm(out)

        # 后续层（带残差）
        for lstm in self.lstm_layers:
            residual = out
            out, _ = lstm(out)
            out = self.dropout(out)
            out = self.layer_norm(out + residual)  # 残差连接

        # 取最后时间步
        out = out[:, -1, :]

        # 输出
        out = self.fc(out)
        return out


def create_sequences(
    data: np.ndarray,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时间序列样本

    Args:
        data: 原始数据 (n_samples,)
        seq_length: 序列长度

    Returns:
        (X, y) 输入序列和目标值
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def prepare_data(
    data: np.ndarray,
    seq_length: int,
    train_ratio: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备训练和测试数据

    Args:
        data: 原始数据
        seq_length: 序列长度
        train_ratio: 训练集比例

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    # 创建序列
    X, y = create_sequences(data, seq_length)

    # 划分训练测试集
    train_size = int(len(X) * train_ratio)

    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    X_test = torch.FloatTensor(X[train_size:])
    y_test = torch.FloatTensor(y[train_size:])

    # 添加特征维度
    X_train = X_train.unsqueeze(-1)  # (n, seq_len, 1)
    X_test = X_test.unsqueeze(-1)

    return X_train, y_train, X_test, y_test
