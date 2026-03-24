"""
LSTM 模型训练模块
Model training utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
import json

from .lstm_model import LSTMModel, prepare_data
from ..utils.logger import get_default_logger

logger = get_default_logger()


class LSTMTrainer:
    """LSTM模型训练器"""

    def __init__(
        self,
        model: LSTMModel,
        learning_rate: float = 0.001,
        device: str = 'cuda',
        model_save_path: str = "data/models"
    ):
        """
        初始化训练器

        Args:
            model: LSTM模型
            learning_rate: 学习率
            device: 训练设备
            model_save_path: 模型保存路径
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证损失
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, _ = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            X_train: 训练输入
            y_train: 训练目标
            X_val: 验证输入
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping_patience: 早停耐心值
            verbose: 是否显示进度

        Returns:
            训练历史
        """
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0

        # 训练循环
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # 学习率调整
                self.scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.save_model("best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"早停于第 {epoch + 1} 轮")
                        break

                if verbose:
                    iterator.set_postfix({
                        'train_loss': f'{train_loss:.6f}',
                        'val_loss': f'{val_loss:.6f}'
                    })
            else:
                if verbose:
                    iterator.set_postfix({'train_loss': f'{train_loss:.6f}'})

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

        logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        return self.history

    def save_model(self, filename: str = "lstm_model.pth"):
        """
        保存模型

        Args:
            filename: 文件名
        """
        save_path = self.model_save_path / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.output_size,
                'bidirectional': self.model.bidirectional
            }
        }, save_path)
        logger.info(f"模型已保存: {save_path}")

    def load_model(self, filename: str = "best_model.pth"):
        """
        加载模型

        Args:
            filename: 文件名
        """
        load_path = self.model_save_path / filename
        if not load_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"模型已加载: {load_path}")

    def save_training_history(self, filename: str = "training_history.json"):
        """保存训练历史"""
        save_path = self.model_save_path / filename
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存: {save_path}")


def train_from_data(
    data: np.ndarray,
    seq_length: int = 10,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    model_save_path: str = "data/models"
) -> Tuple[LSTMModel, LSTMTrainer, Dict[str, List[float]]]:
    """
    从数据训练模型的便捷函数

    Args:
        data: 时间序列数据
        seq_length: 序列长度
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备
        model_save_path: 模型保存路径

    Returns:
        (模型, 训练器, 训练历史)
    """
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data(data, seq_length)

    # 创建模型
    model = LSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1
    )

    # 创建训练器
    trainer = LSTMTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device,
        model_save_path=model_save_path
    )

    # 训练
    history = trainer.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=epochs,
        batch_size=batch_size
    )

    return model, trainer, history
