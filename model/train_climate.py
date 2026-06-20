import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model.rnn_backbone import ClimateRNN

# 自动适配设备：有GPU用GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def train_one_epoch(model, dataloader, criterion, optimizer):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        pred = model(x)
        # 计算损失
        loss = criterion(pred, y)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def val_one_epoch(model, dataloader, criterion):
    """单轮验证"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():  # 验证阶段不计算梯度，节省显存
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def train_single_model(model, train_loader, val_loader, epochs=20, lr=2e-4, save_path=None):
    """
    训练单个模型，返回训练/验证损失历史/最优轮数/最优验证损失
    """
    criterion = nn.MSELoss()
    # 降低学习率 + 权重衰减抑制过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_epoch = 0  # 记录最优模型对应的轮数，方便后续分析

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = val_one_epoch(model, val_loader, criterion)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ========== 验证loss下降时保存最优权重 ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            if save_path:
                torch.save(model.state_dict(), save_path)

    # 训练结束后加载最优权重，用于后续测试
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    print(f"训练完成 | 最优验证Loss: {best_val_loss:.4f} | 出现在第 {best_epoch} 轮")
    return train_loss_history, val_loss_history, best_epoch, best_val_loss


def train_all_models(train_dataset, val_dataset, test_dataset, feature_num, epochs=20, batch_size=32):
    """
    依次训练 RNN / GRU / LSTM 三个模型，对比结果
    返回: 三个模型的损失历史、模型列表
    """
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_configs = [
        ("SimpleRNN", "rnn", "./results/model_weights/rnn_climate.pth"),
        ("GRU", "gru", "./results/model_weights/gru_climate.pth"),
        ("LSTM", "lstm", "./results/model_weights/lstm_climate.pth")
    ]

    history_dict = {}
    model_dict = {}
    result_table = []  # 用于汇总对比结果

    for name, mode, save_path in model_configs:
        print("\n" + "="*50)
        print(f"开始训练 {name} 模型")
        print("="*50)
        
        model = ClimateRNN(input_size=feature_num, mode=mode).to(device)
        train_hist, val_hist, best_epoch, best_val_loss = train_single_model(
            model, train_loader, val_loader, 
            epochs=epochs, save_path=save_path
        )
        
        history_dict[name] = {"train": train_hist, "val": val_hist}
        model_dict[name] = model

        # 测试集评估（用最优权重）
        test_loss = val_one_epoch(model, test_loader, nn.MSELoss())
        print(f"{name} 测试集 MSE: {test_loss:.4f}")
        
        # 汇总结果
        result_table.append({
            "模型": name,
            "最优轮数": best_epoch,
            "验证集最优MSE": round(best_val_loss, 4),
            "测试集MSE": round(test_loss, 4)
        })

    # 打印对比汇总表
    print("\n" + "="*50)
    print("三模型性能对比汇总：")
    print("-"*50)
    for res in result_table:
        print(f"{res['模型']:10s} | 最优轮数: {res['最优轮数']:2d} | 验证MSE: {res['验证集最优MSE']:.4f} | 测试MSE: {res['测试集MSE']:.4f}")
    print("="*50)

    # 绘制损失对比图（20轮完整曲线，更美观）
    plt.figure(figsize=(10, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, hist) in enumerate(history_dict.items()):
        plt.plot(hist['train'], color=colors[i], linestyle='-', label=f"{name} Train Loss")
        plt.plot(hist['val'], color=colors[i], linestyle='--', label=f"{name} Val Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("RNN / GRU / LSTM Loss Comparison (20 Epochs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/climate_fig/model_loss_compare.png", dpi=200)
    plt.close()

    return history_dict, model_dict, result_table