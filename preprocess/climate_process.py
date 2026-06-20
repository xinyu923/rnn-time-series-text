import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 自动创建结果目录
os.makedirs("./results/climate_fig", exist_ok=True)
os.makedirs("./results/model_weights", exist_ok=True)


def load_raw_climate_data(csv_path):
    """
    加载原始气温数据，间隔6行采样（10分钟→1小时粒度），解析时间戳，绘制基础特征曲线
  
    """
    # 读取csv，从第5行开始每6行取1条，将10分钟粒度降为1小时粒度
    df = pd.read_csv(csv_path)
    df = df[5::6].reset_index(drop=True)
    
    # 解析时间戳列，转为datetime格式
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    
    # 打印数据预览与统计信息
    print("="*50)
    print("原始数据预览：")
    print(df.head())
    print("\n数据统计特性：")
    print(df.describe().transpose())
    print("="*50)

    # 绘制全量特征时序曲线
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features = df[plot_cols].copy()
    plot_features.index = date_time
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i, col in enumerate(plot_cols):
        axes[i].plot(plot_features[col])
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.savefig("./results/climate_fig/raw_feature_all.png", dpi=200)
    plt.close()

    # 绘制前480小时（20天）局部放大曲线
    plot_features_short = df[plot_cols][:480].copy()
    plot_features_short.index = date_time[:480]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i, col in enumerate(plot_cols):
        axes[i].plot(plot_features_short[col])
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.savefig("./results/climate_fig/raw_feature_short.png", dpi=200)
    plt.close()

    return df, date_time


def fill_wind_missing(df):
    """填充风速缺失值：异常值-9999.0替换为0"""
    # 处理平均风速
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    # 处理最大风速
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    return df


def wind_deg_to_xy(df):
    """
    将风向角度+风速分解为X/Y二维矢量，解决角度周期性突变问题
    匹配PPT风向预处理逻辑
    """
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    # 角度转弧度
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    
    # 计算风速的x、y分量
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)
    
    # 绘制风速分量二维直方图
    plt.figure(figsize=(6, 5))
    plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel("Wind X [m/s]")
    plt.ylabel("Wind Y [m/s]")
    plt.tight_layout()
    plt.savefig("./results/climate_fig/wind_xy_dist.png", dpi=200)
    plt.close()

    return df


def add_time_cycle_feature(df, date_time):
    """添加日周期、年周期的正弦/余弦编码，让模型学习时间周期性规律"""
    # 时间戳转秒级时间戳
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60          # 一天的秒数
    year = 365.2425 * day       # 一年的秒数
    
    # 日周期特征
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    # 年周期特征
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    # 绘制单日周期曲线
    plt.figure(figsize=(6, 4))
    plt.plot(np.array(df['Day sin'])[:25], label='Day sin')
    plt.plot(np.array(df['Day cos'])[:25], label='Day cos')
    plt.xlabel("Time [h]")
    plt.title("Time of day signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./results/climate_fig/day_cycle.png", dpi=200)
    plt.close()

    return df, timestamp_s


def temp_fft_analysis(df):
    """对气温做FFT频谱分析，验证日周期、年周期规律，替换原TensorFlow实现"""
    temp = df['T (degC)'].values
    # numpy实现实数FFT
    fft = np.fft.rfft(temp)
    n_samples = len(temp)
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples / hours_per_year
    f_per_year = np.arange(len(fft)) / years_per_dataset

    plt.figure(figsize=(7, 4))
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    plt.xlabel("Frequency (log scale)")
    plt.title("Temperature FFT Spectrum")
    plt.tight_layout()
    plt.savefig("./results/climate_fig/temp_fft.png", dpi=200)
    plt.close()


def split_and_norm_dataset(df):
    """
    数据集划分：训练70%、验证20%、测试10%
    仅用训练集统计量做标准化，防止数据泄露
    """
    data = df.values.astype(np.float32)
    n = len(data)
    
    # 按比例划分
    train_data = data[:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]
    
    # 计算训练集均值、标准差
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    
    # 标准化
    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std