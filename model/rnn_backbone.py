import torch
import torch.nn as nn


class ClimateRNN(nn.Module):
    """
    气温预测通用循环神经网络，支持切换 SimpleRNN / GRU / LSTM
    输入形状: (batch, seq_len, feature_num)
    输出形状: (batch, horizon)  未来horizon小时的气温预测值
    加入Dropout抑制过拟合
    """
    def __init__(self, input_size, hidden_size=64, output_size=168, mode='lstm', num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode

        # 循环层：单层网络的dropout参数不生效，放在全连接层实现
        if mode == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif mode == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif mode == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError("mode must be one of ['rnn', 'gru', 'lstm']")

        # 全连接输出头：加入Dropout层，随机失活部分神经元，抑制过拟合
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),  # 新增Dropout
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # rnn输出: output(所有时间步输出), hidden(最后时间步隐藏态)
        if self.mode == 'lstm':
            output, (h_n, c_n) = self.rnn(x)
        else:
            output, h_n = self.rnn(x)
        
        # 取最后一个时间步的输出，映射为预测结果
        last_out = output[:, -1, :]  # shape: (batch, hidden_size)
        pred = self.fc(last_out)     # shape: (batch, output_size)
        return pred


class TextLSTM(nn.Module):
    """
    字符级文本生成LSTM模型
    输入形状: (batch, seq_len)
    输出形状: (batch, seq_len, vocab_size) 每个位置的字符概率logits
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层，返回所有时间步输出
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # 输出层：映射到词表大小
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # shape: (batch, seq_len, embedding_dim)
        
        # 前向传播，支持传入隐藏态用于生成时的状态延续
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        
        logits = self.fc(output)  # shape: (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐藏态，用于生成阶段"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)