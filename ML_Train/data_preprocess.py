import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 載入X_norm和Y數據，並進行初步檢查
def load_data(data_path="./data/"):
    try:
        X_norm = np.load(f"{data_path}X_norm.npy")
        Y = np.load(f"{data_path}Y.npy")
        print("資料載入成功。")

    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案{e}，請確認檔案路徑是否正確。")
        raise

    # 檢查資料形狀
    print(f"\nX_norm形狀 (樣本數、特徵數)：{X_norm.shape}")
    print(f"Y形狀 (標籤數)：{Y.shape}")

    # 檢查分類標籤的分布情況
    unique_labels, counts = np.unique(Y, return_counts=True)
    print(f"\n標籤類別 (Y)：{unique_labels}")
    print(f"各類別樣本計數：\n0 (敏感/Susceptible)：{counts[0]}個\n1 (耐藥/Resistant)：{counts[1]}個")
    
    return X_norm, Y

# 切分訓練集、進行Z-score標準化
def preprocess_data(X_norm, Y, test_size=0.2, random_state=42):
    # 1. 切割資料
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_norm, Y, test_size=test_size, random_state=random_state, stratify=Y
    )

    # 2. 實作 Z-score 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("\n資料切割與標準化完成")
    print(f"訓練集形狀：{X_train_scaled.shape}")
    print(f"測試集形狀：{X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test