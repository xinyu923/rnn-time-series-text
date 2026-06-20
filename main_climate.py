from preprocess.climate_process import (
    load_raw_climate_data, fill_wind_missing, wind_deg_to_xy,
    add_time_cycle_feature, temp_fft_analysis, split_and_norm_dataset
)
from utils.datasets import ClimateDataset
from model.train_climate import train_all_models


if __name__ == "__main__":
    csv_path = "./data/jena_climate_2009_2016.csv"
    
    # ========== 步骤1：全套数据预处理 ==========
    print(">>> 开始数据加载与预处理")
    df, date_time = load_raw_climate_data(csv_path)
    df = fill_wind_missing(df)
    df = wind_deg_to_xy(df)
    df, timestamp_s = add_time_cycle_feature(df, date_time)
    temp_fft_analysis(df)
    
    # 数据集划分与标准化
    train_data, val_data, test_data, train_mean, train_std = split_and_norm_dataset(df)
    feature_num = train_data.shape[-1]
    print(f"特征维度: {feature_num}")
    print(f"训练集长度: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # ========== 步骤2：构建PyTorch数据集 ==========
    lookback = 720   # 历史30天
    horizon = 168    # 预测7天
    
    train_dataset = ClimateDataset(train_data, lookback=lookback, horizon=horizon)
    val_dataset = ClimateDataset(val_data, lookback=lookback, horizon=horizon)
    test_dataset = ClimateDataset(test_data, lookback=lookback, horizon=horizon)
    
    print(f"训练样本数: {len(train_dataset)}")

    # ========== 步骤3：三模型对比训练 ==========
    print(">>> 开始三模型对比训练")
    history_dict, model_dict, result_table = train_all_models(
    train_dataset, val_dataset, test_dataset,
    feature_num=feature_num, epochs=20, batch_size=32
)

    print("\n" + "="*50)
    print("气温预测实验完成，结果已保存至 results/climate_fig/")
    print("="*50)