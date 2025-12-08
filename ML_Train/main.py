# src/main.py
import data_preprocess
import model
import os

# 檢查路徑是否存在，不存在就自動創建
def create_dirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"已新建資料夾: {path}")
        else:
            print(f"資料夾已存在: {path}")

def run_ml_pipeline():
    # 0. 創建結果資料夾
    MODELS_PATH = "./results/models/"
    FIGURES_PATH = "./results/figures/"

    print("--- 檢查與創建結果資料夾 ---")
    create_dirs([MODELS_PATH, FIGURES_PATH])

    # 1. 數據處理與預處理
    print("--- 步驟 1: 數據載入與預處理 ---")
    X_norm, Y = data_preprocess.load_data()
    X_train_scaled, X_test_scaled, Y_train, Y_test = data_preprocess.preprocess_data(X_norm, Y)
    
    # 2. 模型訓練
    print("\n--- 步驟 2: 模型訓練 ---")
    model_lr, model_rf, model_svm = model.train_models(X_train_scaled, Y_train)
    
    # 3. 性能評估
    print("\n--- 步驟 3: 統計推論與性能評估 ---")
    model.evaluate_model(model_lr, X_test_scaled, Y_test, "Logistic Regression")
    model.evaluate_model(model_rf, X_test_scaled, Y_test, "Random Forest (Tuned)")
    model.evaluate_model(model_svm, X_test_scaled, Y_test, "SVM RBF Kernel")
    
    # 4. 可解釋性分析
    print("\n--- 步驟 4: 可解釋性分析 ---")
    
    # A. RF 特徵重要性
    model.feature_importance_analysis(model_rf)
    
    # B. SVM 支持向量分析
    model.support_vector_analysis(model_svm, X_train_scaled, Y_train)
    
    # C. ROC 曲線比較 (綜合性能)
    models = {
        "LR": model_lr,
        "RF Tuned": model_rf,
        "SVM RBF": model_svm
    }
    model.plot_roc_comparison(models, X_test_scaled, Y_test)
    
    print("\n所有傳統機器學習流程已執行完畢。請查看 results/figures/ 資料夾獲取圖表。")

if __name__ == "__main__":
    run_ml_pipeline()