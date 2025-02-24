import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. 数据读取及预处理
# ----------------------------
file_path = 'C:/Users/Administrator/Desktop/vandy3rd/Project/CR/Cytokine analysis for revision sinus surgery/Sep2024respositorydeID2.xlsx'
data = pd.read_excel(file_path, sheet_name='out-s1')

# 删除 'prior_surgery' 缺失的样本
data_clean = data.dropna(subset=['prior_surgery'])

# 任务一：使用 IL1a 到 GMCSF 作为特征，考虑 prior_surgery 的影响
# 此处去掉 'phenotype'、'prior_surgery' 以及 'number_prior_surgeries' 列，其余列均作为特征
X_task1 = data_clean.drop(columns=['phenotype', 'prior_surgery', 'number_prior_surgeries'])
y_task1 = data_clean['prior_surgery']  # 假设为二分类变量（例如 0 与 1）

# 填充缺失值：用各列均值替换缺失值
X_task1 = X_task1.apply(lambda col: col.fillna(col.mean()), axis=0)

# 标准化特征数据
scaler = StandardScaler()
X_task1 = pd.DataFrame(scaler.fit_transform(X_task1), columns=X_task1.columns)

# 保存特征名称列表（用于后续绘图）
features_task1 = X_task1.columns.tolist()

# ----------------------------
# 2. 使用 Leave-One-Out 交叉验证
# ----------------------------
loo = LeaveOneOut()

# 用于保存每次迭代的真实标签、预测标签、完整预测概率和特征重要性
y_true_all = []
y_pred_all = []
y_prob_all = []  # 存储完整的概率输出，每个元素为长度为 2 的数组
feature_importances_list = []

for train_index, test_index in loo.split(X_task1):
    X_train, X_test = X_task1.iloc[train_index], X_task1.iloc[test_index]
    y_train, y_test = y_task1.iloc[train_index], y_task1.iloc[test_index]
    
    # 初始化并训练随机森林分类器
    forest_clf = RandomForestClassifier(random_state=0)
    forest_clf.fit(X_train, y_train)
    
    # 保存当前模型的特征重要性
    feature_importances_list.append(forest_clf.feature_importances_)
    
    # 对测试样本进行预测
    y_pred = forest_clf.predict(X_test)
    # 保存完整的预测概率（数组形状为 (2,)）
    y_prob = forest_clf.predict_proba(X_test)[0]
    
    # 保存当前样本的真实标签、预测标签和预测概率
    y_true_all.append(y_test.values[0])
    y_pred_all.append(y_pred[0])
    y_prob_all.append(y_prob)

# ----------------------------
# 3. 计算总体性能指标
# ----------------------------
# 计算 F1 分数
f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted')

# 为计算 AUC，我们需要将真实标签二值化，并使用完整的概率输出
# 由于是二分类，构造一个 (n_samples, 2) 的标签矩阵，其中每行：[1 - y, y]
y_true_all = np.array(y_true_all)
y_true_all_binarized = np.vstack((1 - y_true_all, y_true_all)).T  # 每行：[概率为类别0, 概率为类别1]
y_prob_all = np.array(y_prob_all)  # 形状为 (n_samples, 2)

# 计算 AUC：对二分类问题，用 multi_class='ovr' 计算 macro 和 weighted
auc_macro = roc_auc_score(y_true_all_binarized, y_prob_all, average='macro', multi_class='ovr')
auc_weighted = roc_auc_score(y_true_all_binarized, y_prob_all, average='weighted', multi_class='ovr')

# 混淆矩阵
conf_matrix = confusion_matrix(y_true_all, y_pred_all)

print("Leave-One-Out Cross-Validation Metrics:")
print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"AUC (macro): {auc_macro:.4f}")
print(f"AUC (weighted): {auc_weighted:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# ----------------------------
# 4. 计算特征重要性均值和标准差
# ----------------------------
importances_array = np.array(feature_importances_list)  # 形状为 (n_samples, n_features)
mean_importances = np.mean(importances_array, axis=0)
std_importances = np.std(importances_array, axis=0)

# 按均值重要性从高到低排序
sorted_indices = np.argsort(mean_importances)[::-1]
sorted_mean = mean_importances[sorted_indices]
sorted_std = std_importances[sorted_indices]
sorted_features = [features_task1[i] for i in sorted_indices]

# ----------------------------
# 5. 绘制特征重要性柱状图（带误差条）
# 每 40 个特征绘制一张图
# ----------------------------
num_features = len(sorted_mean)
num_per_plot = 45
num_plots = int(np.ceil(num_features / num_per_plot))

for i in range(num_plots):
    start = i * num_per_plot
    end = min((i + 1) * num_per_plot, num_features)
    
    chunk_mean = sorted_mean[start:end]
    chunk_std = sorted_std[start:end]
    chunk_features = sorted_features[start:end]
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances (Best Features {start+1} to {end})")
    y_positions = np.arange(len(chunk_mean))
    # 绘制横向柱状图，xerr 显示标准差
    plt.barh(y_positions, chunk_mean, xerr=chunk_std, align='center', color='skyblue', ecolor='black')
    plt.yticks(y_positions, chunk_features)
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()  # 将最重要的特征显示在顶部
    plt.tight_layout()
    plt.show()

