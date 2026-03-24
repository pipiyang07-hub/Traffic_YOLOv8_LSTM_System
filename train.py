"""
LSTM 流量预测模型训练脚本
Training script for LSTM traffic prediction model

Usage:
    python train.py --data data/traffic_data.csv
    python train.py --data data/traffic_data.csv --epochs 100 --hidden-size 64
"""

import argparse
from pathlib import Path
import json

from src.prediction.lstm_model import LSTMModel, prepare_data
from src.prediction.trainer import LSTMTrainer
from src.utils.data_utils import load_config, load_traffic_data, normalize_data
from src.utils.logger import setup_logger

logger = setup_logger(log_file="logs/train.log")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LSTM流量预测模型训练'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='流量数据文件路径'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='模型输出路径'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=None,
        help='序列长度'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='隐藏层大小'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='LSTM层数'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='批次大小'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=None,
        help='学习率'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='训练设备'
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 合并命令行参数
    pred_config = config['prediction']
    seq_length = args.seq_length or pred_config['sequence_length']
    hidden_size = args.hidden_size or pred_config['hidden_size']
    num_layers = args.num_layers or pred_config['num_layers']
    epochs = args.epochs or pred_config['epochs']
    batch_size = args.batch_size or pred_config['batch_size']
    learning_rate = args.learning_rate or pred_config['learning_rate']
    dropout = pred_config['dropout']

    # 加载数据
    logger.info(f"加载数据: {args.data}")
    df = load_traffic_data(args.data)

    # 准备时间序列数据
    logger.info("准备时间序列数据...")
    # 按时间聚合
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
    time_series = df.groupby('timestamp')['count'].sum()

    # 归一化
    data = time_series.values.astype('float32')
    normalized_data, norm_params = normalize_data(data, method='minmax')

    # 创建序列数据
    X_train, y_train, X_test, y_test = prepare_data(
        normalized_data.values,
        seq_length
    )

    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 创建模型
    logger.info("创建LSTM模型...")
    model = LSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    )

    logger.info(f"模型结构: {model}")

    # 创建训练器
    model_save_path = args.output or config['paths']['models']
    trainer = LSTMTrainer(
        model=model,
        learning_rate=learning_rate,
        device=args.device,
        model_save_path=model_save_path
    )

    # 训练模型
    logger.info("开始训练...")
    history = trainer.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=15,
        verbose=True
    )

    # 保存模型和归一化参数
    trainer.save_model("best_model.pth")
    trainer.save_training_history()

    # 保存归一化参数
    norm_params_path = Path(model_save_path) / "norm_params.json"
    with open(norm_params_path, 'w') as f:
        json.dump({
            'min': float(norm_params['min']),
            'max': float(norm_params['max']),
            'method': 'minmax'
        }, f, indent=2)

    logger.info(f"归一化参数已保存: {norm_params_path}")

    # 输出结果
    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {min(history['val_loss']):.6f}")
    logger.info(f"模型保存路径: {model_save_path}")
    logger.info("=" * 50)

    print(f"\n训练完成！")
    print(f"最佳验证损失: {min(history['val_loss']):.6f}")
    print(f"模型保存路径: {model_save_path}")


if __name__ == "__main__":
    main()
