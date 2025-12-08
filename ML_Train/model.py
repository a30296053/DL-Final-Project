from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

TARGET_NAMES = ['Susceptible (0)', 'Resistant (1)']
RESULTS_PATH_F = "./results/figures/"
RESULTS_PATH_M = "./results/models/"

# 訓練模型(LR, RF 和 SVM)
def train_models(X_train, Y_train):
    # 1. Logistic Regression (llinear baseline)
    model_lr = LogisticRegression(C=1.0, random_state=42, solver="liblinear")
    print("\n開始訓練 Logistic Regression model...")
    model_lr.fit(X_train, Y_train)
    
    # 2. Random Forest (有調過參數)
    model_rf = RandomForestClassifier(n_estimators=250, criterion="gini", max_depth=20, random_state=42, n_jobs=-1)
    print("開始訓練 Random Forest Classifier...")
    model_rf.fit(X_train, Y_train)
    
    # 3. SVM (RBF Kernel, 設置 probability=True 以便計算 ROC 曲線)
    model_svm = SVC(kernel="rbf", C=1.0, random_state=42, probability=True)
    print("開始訓練 SVM model...")
    model_svm.fit(X_train, Y_train)

    # 4. 模型儲存
    joblib.dump(model_rf, f'{RESULTS_PATH_M}rf_tuned_model.pkl')
    joblib.dump(model_svm, f'{RESULTS_PATH_M}svm_rbf_model.pkl')
    joblib.dump(model_lr, f'{RESULTS_PATH_M}lr_baseline_model.pkl')
    
    # 載入模型
    # loaded_rf = joblib.load('results/models/rf_tuned_model.pkl')

    print("所有模型訓練完成並已儲存至 results/models/。")
    return model_lr, model_rf, model_svm

# 驗證模型(計算並輸出單一模型的分類報告)
def evaluate_model(model, X_test, Y_test, model_name):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names=TARGET_NAMES)

    print(f"\n--- {model_name} 評估結果 ---")
    print(f"整體準確度 (Overall Accuracy)：{accuracy:.4f}")
    print("\n分類報告 (Classification Report)：\n")
    print(report)


# 可解釋性 / 視覺化函數
def plot_roc_comparison(models, X_test, Y_test):
    plt.figure(figsize=(10, 8))
    print("\n--- 模型性能 AUC 總結 ---")

    for name, model in models.items():
        Y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, Y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        print(f"{name} AUC: {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.title("ROC Curve Comparison for Anti-Drug Resistance Prediction")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH_F}model_roc_comparison.png")
    print(f"ROC 曲線比較圖表已儲存至 {RESULTS_PATH_F}model_roc_comparison.png")
    plt.close()

# RF 特徵重要性分析與視覺化
def feature_importance_analysis(model_rf):
    importances = model_rf.feature_importances_
    top_10_indices = np.argsort(importances)[::-1][:10]
    top_10_scores = importances[top_10_indices]

    df_importance = pd.DataFrame({
        "Feature": [f"Feature Index {i}" for i in top_10_indices], 
        "Importance": top_10_scores
    }).sort_values(by="Importance", ascending=True)

    # 繪製圖表
    plt.figure(figsize=(10, 6))
    plt.barh(df_importance["Feature"], df_importance["Importance"], color="darkred")
    plt.xlabel("Feature Importance Scores")
    plt.title("Top 10 Most Important Features")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH_F}top_10_feature_importance_final.png")
    print(f"\nRF 特徵重要性圖表已儲存至 {RESULTS_PATH_F}top_10_feature_importance_final.png")

    # 輸出表格
    print("\n前 10 個最重要的特徵索引和分數:")
    print(df_importance[['Feature', 'Importance']].sort_values(by='Importance', ascending=False).to_string(index=False))
    plt.close()

# SVM 支持向量分析與 PCA 視覺化
def support_vector_analysis(model_svm, X_train_scaled, Y_train):
    # 降維處理 (PCA)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # 提取支持向量索引
    support_vector_indices_in_train = model_svm.support_
    num_support_vectors = len(support_vector_indices_in_train)

    # 繪製支持向量圖表
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, 
        cmap=plt.cm.coolwarm, marker=".", alpha=0.3
    )

    svs = X_train_pca[support_vector_indices_in_train]
    plt.scatter(
        svs[:, 0], svs[:, 1], color="none", edgecolors="black", marker="o", 
        s=150, alpha=0.6, linewidth=1.5, label=f"Support Vectors ({num_support_vectors} pts)"
    )

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.title('SVM Support Vectors Visualization (Projected onto 2D PCA Space)')
    plt.legend(handles=scatter.legend_elements()[0], labels=TARGET_NAMES, title="Classes")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH_F}svm_support_vectors_pca.png")
    print(f"\nSVM 支持向量總數: {num_support_vectors} 個，佔訓練集比例: {num_support_vectors / len(X_train_scaled):.2%}")
    print(f"SVM 支持向量視覺化圖表已儲存至 {RESULTS_PATH_F}svm_support_vectors_pca.png")
    plt.close()