# RNN PyTorch 实验

本项目基于 PyTorch 实现了两个循环神经网络任务：

1. 耶拿气候数据集的多步温度预测（使用 SimpleRNN/GRU/LSTM 进行对比实验）
2. 莎士比亚风格剧本生成（基于 LSTM 的字符级文本生成）

---

## 项目概述

该仓库包含完整的数据预处理、模型训练与结果保存流程，方便复现实验与对比不同循环神经网络结构的性能。

- `main_climate.py`：气候预测任务一键运行入口
- `main_text.py`：文本生成任务一键运行入口
- `preprocess/`：数据预处理模块
- `model/`：RNN 模型与训练逻辑
- `utils/`：数据集与辅助工具
- `data/`：原始数据集
- `results/`：训练结果与生成样本输出

---

## 任务说明

### 1. 气候预测

使用耶拿气候数据集进行多步温度预测。项目中实现了三个循环网络结构：

- `SimpleRNN`
- `GRU`
- `LSTM`

实验流程：

- 加载并处理原始气候数据
- 补全风向缺失值并转换风向向量特征
- 添加时间周期特征
- 划分训练/验证/测试集并标准化
- 使用 PyTorch Dataset 封装序列数据
- 训练对比不同网络并保存预测结果

### 2. 文本生成

使用莎士比亚文本进行字符级生成模型训练。核心设计包括：

- 构建字符索引表
- 将文本转为整数序列
- 训练 LSTM 生成模型
- 进行测试集困惑度评估（PPL）
- 生成莎士比亚风格文本并保存结果

---

## 目录结构

```text
.
├── data/
│   ├── jena_climate_2009_2016.csv
│   └── shakespeare.txt
├── preprocess/
│   ├── climate_process.py
│   └── text_process.py
├── model/
│   ├── rnn_backbone.py
│   ├── train_climate.py
│   └── train_text_gen.py
├── utils/
│   └── datasets.py
├── results/
│   ├── climate_fig/
│   ├── text_sample/
│   └── model_weights/
├── main_climate.py
├── main_text.py
└── requirements.txt
```

---

## 环境与依赖

建议使用 Python 3.9，并安装以下依赖：

```text
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
torchvision==0.18.1+cu121
matplotlib==3.9.4
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.6.1
```

如果需要，可使用 `requirements.txt` 创建环境：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 运行气候预测

```bash
python main_climate.py
```

该脚本会完成数据预处理、划分数据集、构建 `ClimateDataset`，并进行 SimpleRNN/GRU/LSTM 的训练与对比，最终将结果保存到 `results/climate_fig/`。

### 运行文本生成

```bash
python main_text.py
```

该脚本会构建字符级训练数据、训练 LSTM 模型、计算测试集困惑度，并生成莎士比亚风格文本，保存到 `results/text_sample/generate_play.txt`。

---

## 结果存储

- `results/climate_fig/`：气候预测结果可视化图表
- `results/model_weights/`：训练模型权重
- `results/text_sample/`：生成的文本样本

---

## 说明

- 本项目适合用于学习循环神经网络（RNN、GRU、LSTM）在时间序列预测与文本生成任务中的应用。
- 如果本地显存较小，可在 `main_text.py` 中调整 `batch_size` 或滑动步长 `step` 参数以降低内存占用。

欢迎将本仓库用于教学、实验和模型对比研究。
