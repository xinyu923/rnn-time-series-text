from preprocess.text_process import load_shakespeare_txt
from utils.datasets import TextDataset
from model.train_text_gen import train_text_model, generate_text, evaluate_model_ppl
from torch.utils.data import DataLoader


if __name__ == "__main__":
    txt_path = "./data/shakespeare.txt"
    
    # 预处理：构建词表，文本转索引
    text_int, char2idx, idx2char, vocab_size, seq_len = load_shakespeare_txt(txt_path)
    
    # ========== 修改1：划分训练集/测试集（9:1），规范实验，避免数据泄露 ==========
    split_idx = int(len(text_int) * 0.9)
    train_int = text_int[:split_idx]
    test_int = text_int[split_idx:]
    
    # ========== 修改2：滑窗增加步长step=10，样本量直接减少90%，是核心提速点 ==========
    # 相邻样本重叠度大幅降低，不影响模型学习效果，但训练速度提升10倍左右
    train_dataset = TextDataset(train_int, seq_len=seq_len, step=10)
    test_dataset = TextDataset(test_int, seq_len=seq_len, step=10)
    
    # ========== 修改3：调大batch_size，提升GPU利用率，进一步减少迭代次数 ==========
    # 本地显存如果小于4G，可以改回32；6G及以上推荐64
    batch_size = 64
    
    # 训练模型
    print(">>> 开始训练文本生成模型")
    print(f"训练样本数：{len(train_dataset)}，测试样本数：{len(test_dataset)}")
    model = train_text_model(train_dataset, vocab_size, seq_len, epochs=20, batch_size=batch_size)
    
    # ========== 新增：测试集定量评估，计算困惑度PPL ==========
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_ppl = evaluate_model_ppl(model, test_loader, vocab_size)
    print(f"\n测试集平均交叉熵损失: {test_loss:.4f}")
    print(f"测试集困惑度 PPL: {test_ppl:.2f}")
    
    # 生成剧本样例
    start_str = "First Citizen:\nSpeak, speak.\n"
    generated = generate_text(model, start_str, char2idx, idx2char, generate_num=800, temperature=0.7)
    
    print("\n" + "="*50)
    print("生成的莎士比亚风格剧本：")
    print("="*50)
    print(generated)
    
    # 保存生成结果
    with open("./results/text_sample/generate_play.txt", 'w', encoding='utf-8') as f:
        f.write(generated)
    
    print("\n生成结果已保存至 results/text_sample/generate_play.txt")