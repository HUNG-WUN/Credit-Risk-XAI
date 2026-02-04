import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
# 加入 roc_curve 與 auc
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ======================
# 1. 資料載入與預覽
# ======================
try:
    df = pd.read_csv("UCI_Credit_Card.csv")
    print("\n=== [1] Dataset Preview (First 5 Rows) ===")
    print(df.head())
except FileNotFoundError:
    print("錯誤：找不到 UCI_Credit_Card.csv，請確認檔案路徑。")
    exit()
X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

print(f"\n訓練特徵數量：{X.shape[1]}") # 應該會從 24 變成 23

ratio = float(y.value_counts()[0]) / y.value_counts()[1]
print(f"\n自動計算之 Scale_Pos_Weight: {ratio:.2f}")

# ======================
# 2. 模型穩定性驗證 (K-Fold)
# ======================
print("\n=== [2] Model Stability Evaluation (Repeated K-Fold) ===")
# 使用 5-Fold 重複 10 次，共 50 次實驗
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_model = xgb.XGBClassifier(tree_method='hist', device='cuda', scale_pos_weight=ratio, random_state=42)
cv_results = cross_val_score(cv_model, X, y, cv=cv_strategy, scoring='accuracy')

print(f">> Mean Accuracy (平均準確率): {cv_results.mean():.4f}")
print(f">> Std Deviation (標準差/穩定性): {cv_results.std():.4f}")

# ======================
# 3. 訓練與 GPU 監控
# ======================
print("\n=== [3] Training Optimized XGBoost on GPU ===")
os.system('nvidia-smi')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=5,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=ratio,
    eval_metric=['error', 'logloss']
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# ======================
# 4. 績效評估 (含 ROC & AUC)
# ======================
print("\n=== [4] Confusion Matrix, Classification Report & ROC ===")
y_pred = model.predict(X_test)
# 計算預測機率 (ROC 曲線需要)
y_probs = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
# 文字版混淆矩陣
print("\nConfusion Matrix (Text Mode):")
print(f"                Predicted Negative    Predicted Positive")
print(f"Actual Negative:     {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"Actual Positive:     {cm[1][0]:<15} {cm[1][1]:<15}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
# --- 繪製 ROC 曲線並計算 AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print(f"Calculated AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 對角基準線
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("roc_auc_curve.png")
plt.close()

# --- 繪製 Accuracy 折線圖 ---
results = model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot([1-x for x in results['validation_0']['error']], label='Train Accuracy')
plt.plot([1-x for x in results['validation_1']['error']], label='Test Accuracy')
plt.title('Epoch vs Accuracy')
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

# --- 繪製混淆矩陣 ---
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
disp.plot(cmap='Oranges', values_format='d')
plt.savefig("confusion_matrix.png")
plt.close()

# ======================
# 5. SHAP 解釋
# ======================
print("\n=== [5] Generating Comprehensive SHAP Analysis ===")
explainer = shap.TreeExplainer(model)
# 獲取 SHAP Explanation 物件
shap_values_obj = explainer(X_test)
# 傳統 shap_values 陣列 (用於部分舊圖表)
shap_values_arr = explainer.shap_values(X_test)

# 1. SHAP Bar Plot (全局重要性)
plt.figure()
shap.plots.bar(shap_values_obj, show=False)
plt.savefig("shap_1_bar.png", bbox_inches='tight')
plt.close()

# 2. SHAP Summary Plot (Beeswarm 蜂群圖)
plt.figure()
shap.plots.beeswarm(shap_values_obj, show=False)
plt.savefig("shap_2_summary_beeswarm.png", bbox_inches='tight')
plt.close()

# 3. SHAP Waterfall Plot (單一個案解析 - 第1筆)
plt.figure()
shap.plots.waterfall(shap_values_obj[0], show=False)
plt.savefig("shap_3_waterfall.png", bbox_inches='tight')
plt.close()

# 4. SHAP Dependence Plot (觀察 PAY_0 的依賴關係)
plt.figure()
shap.plots.scatter(shap_values_obj[:, "PAY_0"], color=shap_values_obj, show=False)
plt.savefig("shap_4_dependence.png", bbox_inches='tight')
plt.close()

# 5. SHAP Force Plot (力導向圖 - 第1筆)
plt.figure()
shap.force_plot(explainer.expected_value, shap_values_arr[0, :], X_test.iloc[0, :], matplotlib=True, show=False)
plt.savefig("shap_5_force.png", bbox_inches='tight')
plt.close()

# 6. SHAP Decision Plot (決策路徑圖 - 前20筆)
plt.figure()
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)): expected_value = expected_value[1]
shap.decision_plot(expected_value, shap_values_arr[:20], X_test.iloc[:20], show=False)
plt.savefig("shap_6_decision.png", bbox_inches='tight')
plt.close()

# 文字版重要性
vals = np.abs(shap_values_arr).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['Feature', 'Mean_SHAP'])
print("\nTop 10 Features (SHAP):")
print(feature_importance.sort_values(by='Mean_SHAP', ascending=False).head(10).to_string(index=False))

print("\n=== [6] 任務完成！已儲存 完圖表 ===")