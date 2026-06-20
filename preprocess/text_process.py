import numpy as np
import os

os.makedirs("./results/text_sample", exist_ok=True)


def load_shakespeare_txt(txt_path):
    """
    加载莎士比亚文本，构建字符级词表，将文本转为索引数组
    返回: 索引数组、字符→索引映射、索引→字符映射、词表大小、序列长度
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 构建词表：所有出现过的字符去重排序
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    
    # 双向映射
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    # 文本转整数索引
    text_as_int = np.array([char2idx[c] for c in text], dtype=np.int64)
    
    seq_len = 100  # 输入序列长度
    print(f"词表大小: {vocab_size}")
    print(f"文本总长度: {len(text)}")
    
    return text_as_int, char2idx, idx2char, vocab_size, seq_len