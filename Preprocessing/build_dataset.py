import os
from pathlib import Path

import numpy as np
import pandas as pd


# ======= 路徑設定（請改成你自己的） =======
CODES_FILE = r"id\2017\Fusidic acid_data.csv"           # 存 code和 label 的檔案
BINNED_DIR = Path(r"binned_6000\2017")  # 每個 sample 的 6000 維 txt 在這裡

OUTPUT_NPZ = r"fusidic_2017_XY(raw).npz"         # 輸出的壓縮檔


def main():
    df_codes = pd.read_csv(CODES_FILE)

    # 1. 讀 code 列表
    if "code" not in df_codes.columns:
        raise ValueError("codes 檔案裡需要有一欄叫做 'code'")

    codes = df_codes["code"].astype(str).tolist()

    print("********************")
    print(str(BINNED_DIR) +f"/{codes[1]}.txt")


    # 2. 讀 label 欄（R/S）
    if "code" not in df_codes.columns:
        raise ValueError("codes 檔案裡需要有一欄叫做 'code'")
        
    # 將 R/S 轉成 1/0
    label_series = df_codes["label"].astype(str).str.strip().str.upper()
    Y = label_series.map({"R": 1, "S": 0}).to_numpy()

    X_list = []
    used_codes = []
    missing_codes = []

    for code in codes:
        file_path = BINNED_DIR / f"{code}.txt"
        if not file_path.exists():
            missing_codes.append(code)
            continue
        
        # 3. 讀每個 sample 的 6000 維特徵 (分隔是空白)
        df_feat = pd.read_csv(file_path, sep=r"\s+", header=0)

        x = df_feat["binned_intensity"].to_numpy()

        # 確保長度是 6000
        if x.shape[0] != 6000:
            print(f"[警告] {code} 不是 6000 維，實際長度 = {x.shape[0]}，先跳過")
            continue

        X_list.append(x)
        used_codes.append(code)

    if not X_list:
        raise RuntimeError("沒有成功讀取任何 sample，請檢查路徑與檔名。")

    # 4. 堆成 X 矩陣
    X = np.vstack(X_list)   # shape = (n_samples, 6000)
    print("X shape:", X.shape)

    # 對應的 Y 也要同步裁剪到 used_codes (可能X有漏)
    if Y is not None:
        # 建一個 code -> label 的 dict，確保順序一致
        code_to_label = dict(zip(df_codes["code"].astype(str), Y))
        Y_aligned = np.array([code_to_label[c] for c in used_codes])
    else:
        Y_aligned = None

    # 5. 存成 npz
    if Y_aligned is not None:
        np.savez_compressed(OUTPUT_NPZ, X=X, Y=Y_aligned, codes=np.array(used_codes))
    else:
        np.savez_compressed(OUTPUT_NPZ, X=X, codes=np.array(used_codes))

    print(f"已將資料存成 {OUTPUT_NPZ}")



if __name__ == "__main__":
    main()
