import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler

# 1. 读取数据文件
file_path = 'C:/Users/Administrator/Desktop/vandy3rd/Project/CR/Cytokine analysis for revision sinus surgery/Sep2024respositorydeID2.xlsx'
data = pd.read_excel(file_path, sheet_name='out1.3')

# 2. 数据清理：删除 'number_prior_surgeries' 和 'prior_surgery' 中的缺失值
data_clean = data.dropna(subset=['number_prior_surgeries', 'prior_surgery'])

# 3. 将 'number_prior_surgeries' 分为两类：例如，将 1 与大于 1 区分开来
#    此处采用 pd.cut 分箱（注意：根据 bins 与 labels 的设置，这里实际上分为三个类别，
#    如有需要可根据实际情况修改 bins 与 labels）
data_clean['prior_surgery_category'] = pd.cut(data_clean['number_prior_surgeries'],
                                              bins=[-1, 0, 1, np.inf],
                                              labels=[0, 1, 2])

# 4. 任务一：使用 IL1a 到 GMCSF 作为特征，考虑 prior_surgery 的影响
#    假设数据中除 'phenotype', 'prior_surgery', 'number_prior_surgeries', 'prior_surgery_category'
#    外的所有列都是特征
X_task1 = data_clean.drop(columns=['phenotype', 'prior_surgery', 'number_prior_surgeries', 'prior_surgery_category'])
y_task1 = data_clean['prior_surgery_category']

# 5. 填充缺失值：将每列缺失值替换为该列的均值
X_task1 = X_task1.apply(lambda col: col.fillna(col.mean()), axis=0)

# 6. 标准化特征数据
scaler = StandardScaler()
X_task1 = pd.DataFrame(scaler.fit_transform(X_task1), columns=X_task1.columns)

# 7. 使用 LabelEncoder 将目标变量转换为数值编码
le = LabelEncoder()
y_task1_encoded = le.fit_transform(y_task1)

# 8. 使用 Leave-One-Out 交叉验证
loo = LeaveOneOut()

# 用于保存每次折的预测结果、真实标签、预测概率，以及特征重要性
y_true_all = []
y_pred_all = []
y_prob_all = []
importances_list = []

fold = 1
for train_index, test_index in loo.split(X_task1, y_task1_encoded):
    X_train, X_test = X_task1.iloc[train_index], X_task1.iloc[test_index]
    y_train, y_test = y_task1_encoded[train_index], y_task1_encoded[test_index]
    
    # 初始化并训练随机森林模型
    forest_clf = RandomForestClassifier(random_state=0)
    forest_clf.fit(X_train, y_train)
    
    # 保存该折模型的特征重要性
    importances_list.append(forest_clf.feature_importances_)
    
    # 对测试样本进行预测
    y_pred = forest_clf.predict(X_test)
    y_prob = forest_clf.predict_proba(X_test)
    
    # 记录预测结果和真实标签
    y_true_all.append(y_test[0])
    y_pred_all.append(y_pred[0])
    y_prob_all.append(y_prob[0])
    
    fold += 1

# 9. 计算总体性能指标
f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted')

# 为计算 AUC 将真实标签二值化（注意：这里类别为 [0, 1, 2]）
y_true_all_binarized = label_binarize(y_true_all, classes=[0, 1, 2])
auc_macro = roc_auc_score(y_true_all_binarized, y_prob_all, average='macro', multi_class='ovr')
auc_weighted = roc_auc_score(y_true_all_binarized, y_prob_all, average='weighted', multi_class='ovr')

conf_matrix = confusion_matrix(y_true_all, y_pred_all)

print("Leave-One-Out Cross-Validation Metrics:")
print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"AUC (macro): {auc_macro:.4f}")
print(f"AUC (weighted): {auc_weighted:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# 10. 计算每个特征的重要性均值和标准差
importances_array = np.array(importances_list)  # 形状为 (n_folds, n_features)
mean_importances = np.mean(importances_array, axis=0)
std_importances = np.std(importances_array, axis=0)

# 按均值重要性从高到低排序
sorted_indices = np.argsort(mean_importances)[::-1]
sorted_mean = mean_importances[sorted_indices]
sorted_std = std_importances[sorted_indices]
sorted_features = [X_task1.columns[i] for i in sorted_indices]

# 11. 绘制特征重要性图（横向条形图，每 52 个特征绘制一张图，误差条表示标准差）
num_features = len(sorted_mean)
num_per_plot = 4
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
    # 绘制横向条形图，并添加误差条（标准差）
    plt.barh(y_positions, chunk_mean, xerr=chunk_std, align='center', color='skyblue', ecolor='black')
    plt.yticks(y_positions, chunk_features)
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()  # 将最重要的特征显示在顶部
    plt.tight_layout()
    plt.show()