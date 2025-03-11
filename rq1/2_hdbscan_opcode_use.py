import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
import pickle
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import pickle

# 1. 加载 JSON 数据
with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/opcode_embeddings/contract_embeddings_opcodes_chunk.pkl', 'rb') as f:
    contract_embeddings = pickle.load(f)
    
# with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_demo/utils/contract_address_to_index.json', 'r') as f:
#     contract_address_to_index = json.load(f)

# contract_embeddings = [item for item in contract_embeddings if item['address'] in contract_address_to_index]
print("示例数据:", contract_embeddings[0])

# 2. 提取嵌入向量
embeddings = [item['embedding'] for item in contract_embeddings]
X = np.array(embeddings)
print(f"嵌入向量形状: {X.shape}")

# 3. 数据预处理（标准化）
# 根据您的需求，您可以选择是否进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 如果您决定不进行标准化，可以使用原始数据
# X_scaled = X

# 4. 初始化 HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,    # 最小簇大小，根据数据调整
    min_samples=2,          # 用于估计点的密度
    cluster_selection_method='eom',  # 'eom' 或 'leaf'
)

# 5. 执行聚类
labels = clusterer.fit(X_scaled)

with open('hdbscan_label_contract_opcode.pickle','wb') as f:
    pickle.dump(labels,f)

# 6. 获取聚类标签
labels = clusterer.labels_
print(f"聚类标签分布:\n{pd.Series(labels).value_counts()}")

# 7. 将聚类结果添加回数据
for i, item in enumerate(contract_embeddings):
    item['cluster'] = int(labels[i])  # -1 表示噪声点

# 8. 保存聚类结果
with open('./contract_embeddings_with_clusters_hdbscan_opcode.pkl', 'wb') as f:
    pickle.dump(contract_embeddings, f)

# 9. 使用 Pandas 分析聚类结果
df = pd.DataFrame(contract_embeddings)
print("各簇的数量分布:")
print(df['cluster'].value_counts())

# 10. 可选：计算聚类指标
# 由于 HDBSCAN 可能标记噪声点为 -1，通常不包括它们在聚类指标中
mask = labels != -1
if np.sum(mask) > 1:  # 需要至少两个样本
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
    ch_score = calinski_harabasz_score(X_scaled[mask], labels[mask])
    db_score = davies_bouldin_score(X_scaled[mask], labels[mask])
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Index: {ch_score}")
    print(f"Davies-Bouldin Index: {db_score}")
else:
    print("无法计算聚类指标，因为没有足够的聚类。")
    
# 11. 使用 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 12. 创建 DataFrame 包含 t-SNE 结果和聚类标签
df_tsne = pd.DataFrame({
    'tSNE1': X_tsne[:, 0],
    'tSNE2': X_tsne[:, 1],
    'Cluster': labels
})

# 13. 可视化，标记噪声点
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_tsne, 
    x='tSNE1', 
    y='tSNE2', 
    hue='Cluster', 
    palette='tab20', 
    alpha=0.6,
    legend='full'
)
plt.title('HDBSCAN 聚类结果 (t-SNE 降维)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('hdbscan_tsne_opcode.png')
plt.show()
# 11. 降维
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_scaled)

# # 12. 创建 DataFrame 包含 PCA 结果和聚类标签
# df_pca = pd.DataFrame({
#     'PCA1': X_pca[:, 0],
#     'PCA2': X_pca[:, 1],
#     'Cluster': labels
# })

# # 13. 可视化，标记噪声点
# plt.figure(figsize=(12, 8))
# sns.scatterplot(
#     data=df_pca, 
#     x='PCA1', 
#     y='PCA2', 
#     hue='Cluster', 
#     palette='tab20', 
#     alpha=0.6,
#     legend='full'
# )
# plt.title('HDBSCAN 聚类结果 (PCA 降维)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig('hdbscan_pca.png')  # 修正了缺少右括号的问题
# plt.show()
