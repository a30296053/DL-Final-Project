1. 檔案結構：

ml_code/
├── data/	# 數據輸入
│   ├── X_norm.npy		# 原始光譜特徵 (單樣本最大值正規化)
│   └── Y.npy 		# 抗藥性標籤 (0: 敏感, 1: 耐藥)
│
├── data_preprocess.py	# 數據載入、切割、標準化
├── model.py			# 訓練、評估、解釋性分析函式
├── main.py			# 執行入口 (執行整個 ML Pipeline)
│
└── results/	# 輸出結果
    ├── models/                 # 訓練好的模型 (.pkl)
    └── figures/                # 視覺化圖表 (.png)

2. 必要套件：

numpy pandas scikit-learn matplotlib joblib


3. 執行流程：

進入專案資料夾，執行main.py (輸入cd ...\...\ml_code, python main.py 那些的)


4. 結果解釋；

(1)RF 特徵重要性 -> top_10_feature_importance_final.png
鎖定光譜數據中對於抗藥性判斷貢獻最大的特徵索引 (例如 $1399$ 和 $1455$)

(2)SVM 支持向量分析 -> svm_support_vectors_pca.png
視覺化證明了敏感菌株和耐藥菌株在特徵空間中高度重疊（支持向量佔訓練集約 95.73%），解釋了該分類任務的本質困難性

(3)ROC 曲線比較 -> model_roc_comparison.png
綜合比較三種模型的綜合判斷能力，證明 SVM 的表現是最穩健的