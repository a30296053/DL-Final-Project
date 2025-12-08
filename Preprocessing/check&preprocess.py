import numpy as np
import os


npz_path = "fusidic_2017_XY(raw).npz"  # 改成你的檔名

data = np.load(npz_path)

print("=== Keys in NPZ ===")
print(data.files)
print()


"""
if "X" in data.files:
    print("X shape:", data["X"].shape)
    print("X sample:", data["X"][0:3, :10])  # 顯示前 10 個特徵
    print()

if "Y" in data.files:
    print("y shape:", data["Y"].shape)
    print("Y sample (first 20):", data["Y"][:20])
    print("Y unique:", np.unique(data["Y"], return_counts=True))
    print()

if "codes" in data.files:
    print("codes count:", len(data["codes"]))
    print("first 5 codes:", data["codes"][:5])
    print()

"""

# save data with preprocessing    

from sklearn.preprocessing import StandardScaler


X = data["X"]

X_norm = X / X.max(axis=1, keepdims=True)   # 每筆光譜 max=1

np.save('X_norm',X_norm)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

print("X shape:", X_scaled.shape)
print("X sample:", X_scaled[0:3, :10])  # 顯示前 10 個特徵
print()

np.save('X_scaled',X_scaled)

Y = data["Y"]


np.save('Y',Y)