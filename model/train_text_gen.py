import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from model.rnn_backbone import TextLSTM

# 自动适配设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 自动创建结果目录
os.makedirs("./results/model_weights", exist_ok=True)
os.makedirs("./results/text_sample", exist_ok=True)


def train_text_model(train_dataset, vocab_size, seq_len, epochs=20, batch_size=32, lr=1e-3):
    """训练文本生成LSTM模型"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TextLSTM(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_seq, target_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_seq)
            # 交叉熵要求输入形状: (batch*seq_len, vocab_size)，目标: (batch*seq_len)
            loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * input_seq.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "./results/model_weights/text_lstm.pth")
    return model


def evaluate_model_ppl(model, test_loader, vocab_size):
    """
    计算测试集上的困惑度 PPL
    PPL = exp(平均交叉熵损失)，数值越低，语言建模能力越强
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # 测试阶段不计算梯度
        for input_seq, target_seq in test_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            logits, _ = model(input_seq)
            # 计算交叉熵损失，和训练时保持完全一致
            loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            
            total_loss += loss.item() * input_seq.size(0)
            total_samples += input_seq.size(0)
    
    avg_loss = total_loss / total_samples
    ppl = np.exp(avg_loss)  # 困惑度 = 交叉熵损失的自然指数
    return avg_loss, ppl


def generate_text(model, start_string, char2idx, idx2char, generate_num=500, temperature=0.7):
    """
    基于训练好的模型生成文本
    参数:
        start_string: 起始字符串
        temperature: 温度系数，越大越随机，越小越保守
    """
    model.eval()
    # 起始字符串转索引
    input_eval = [char2idx[s] for s in start_string]
    input_eval = torch.tensor(input_eval, dtype=torch.long).unsqueeze(0).to(device)
    
    text_generated = []
    # 初始化隐藏态
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        # 先将起始序列输入模型，得到初始隐藏态
        logits, hidden = model(input_eval, hidden)
        
        for i in range(generate_num):
            # 取最后一个时间步的输出
            last_logits = logits[:, -1, :] / temperature
            # 按概率采样下一个字符（用多项式分布采样）
            probs = torch.softmax(last_logits, dim=-1)
            pred_id = torch.multinomial(probs, num_samples=1).item()
            
            # 将预测的字符加入生成结果
            text_generated.append(idx2char[pred_id])
            
            # 将预测的字符作为下一次的输入，更新隐藏态
            input_eval = torch.tensor([[pred_id]], dtype=torch.long).to(device)
            logits, hidden = model(input_eval, hidden)
    
    return start_string + ''.join(text_generated)