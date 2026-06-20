import torch
from torch.utils.data import Dataset
import numpy as np


class ClimateDataset(Dataset):
    """
    气温预测时序数据集：滑窗生成「历史特征序列 → 未来气温序列」样本对
    """
    def __init__(self, data, lookback=720, horizon=168, target_col=1):
        """
        参数:
            data: 标准化后的numpy数组 (总时长, 特征数)
            lookback: 输入历史序列长度（单位：小时），默认720小时=30天
            horizon: 预测未来序列长度（单位：小时），默认168小时=7天
            target_col: 气温特征所在的列索引，默认第1列是T(degC)
        """
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.target_col = target_col
        # 总样本数 = 总长度 - 历史长度 - 预测长度 + 1
        self.total_samples = len(data) - lookback - horizon + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 输入：idx 到 idx+lookback 的所有特征
        x = self.data[idx : idx + self.lookback, :]
        # 标签：idx+lookback 到 idx+lookback+horizon 的气温列
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, self.target_col]
        
        # 转为torch张量
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class TextDataset(Dataset):
    """
    文本生成字符级数据集：输入字符序列 → 下一个字符序列（错位1位）
    """
    def __init__(self, text_int, seq_len=100, step=10):
        """
        参数:
            text_int: 字符转索引后的一维numpy数组
            seq_len: 输入序列长度
            step: 滑窗步长，值越大样本越少、训练越快，推荐5~20
        """
        self.text_int = text_int
        self.seq_len = seq_len
        self.step = step
        # 重新计算总样本数：按步长跳跃采样
        self.total_samples = (len(text_int) - seq_len) // step

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 按步长计算起始位置
        start = idx * self.step
        # 输入序列：start 到 start+seq_len
        input_seq = self.text_int[start : start + self.seq_len]
        # 目标序列：错位1位
        target_seq = self.text_int[start + 1 : start + self.seq_len + 1]
        
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        return input_seq, target_seq