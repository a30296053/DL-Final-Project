import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import collections

# 讀資料
data = np.load("fusidic_2017_XY(raw).npz")
X = data["X"]            # shape = (n,6000)
Y = data["Y"]            # shape = (n,)
codes = data["codes"]

X_norm = np.load("X_norm.npy")

## bar chart
cnt = collections.Counter(Y)  # y = 0=S, 1=R

plt.figure(figsize=(5,4))
plt.bar(["S (0)", "R (1)"], [cnt[0], cnt[1]], color=["skyblue", "salmon"])
plt.title("Fusidic Acid - R/S Sample Count")
plt.ylabel("Count")
plt.show()


## PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=Y, cmap="coolwarm", alpha=0.55, s=12)
plt.title("PCA of 6000-d MALDI-TOF (Fusidic Acid)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar(label="0=S, 1=R")
plt.show()

# 如果資料太大（>2000）可以抽取部分資料以加速
# 例如：
# idx = np.random.choice(len(X), size=1200, replace=False)
# X_sample = X[idx]
# y_sample = y[idx]


## T_SNE
X_sample = X_norm
Y_sample = Y

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init="random",
    random_state=42
)

X_tsne = tsne.fit_transform(X_sample)


plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=Y_sample, cmap="coolwarm", s=12, alpha=0.75)
plt.title("t-SNE of 6000-d MALDI-TOF (Fusidic Acid)")
plt.colorbar(label="0=S, 1=R")
plt.show()

## mean spectrum
mean_spectrum = X_norm.mean(axis=0)

plt.figure(figsize=(15,4))
plt.plot(mean_spectrum, linewidth=0.8)
plt.title("Mean MALDI-TOF Spectrum (6000 bins)")
plt.xlabel("Bin Index")
plt.ylabel("Intensity")
plt.show()

# R/S spectrum
mean_R = X_norm[Y==1].mean(axis=0)
mean_S = X_norm[Y==0].mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(15,8), sharex=True)

axes[0].plot(mean_R, label="R", alpha=0.7)
axes[0].set_title("Mean Spectrum – R")
axes[0].set_ylabel("Intensity")
axes[0].grid(color = 'k', linestyle = '--', linewidth = 0.5, alpha=0.3)
axes[0].legend()

axes[1].plot(mean_S, label="S", alpha=0.7)
axes[1].set_title("Mean Spectrum – S")
axes[1].set_xlabel("Bin Index")
axes[1].set_ylabel("Intensity")
axes[1].grid(color = 'k', linestyle = '--', linewidth = 0.5, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()


## together
plt.figure(figsize=(15,4))
plt.plot(mean_R, 'r-', label="R", alpha=0.7, linewidth=2)
plt.plot(mean_S, 'b-', label="S", alpha=0.5, linewidth=1.6)
plt.title("Mean Spectrum by Class (R vs S)")
plt.xlabel("Bin Index")
plt.ylabel("Intensity")
plt.grid()
plt.legend()
plt.show()


