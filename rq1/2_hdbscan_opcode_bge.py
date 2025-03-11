from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os
from tqdm import tqdm
import time

data = []
for root, dirs, files in os.walk('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bge_embedding_output/'):
    for file in tqdm(files,'Reading files'):
        if file.endswith(".pickle"):
                # total += 1
                with open(os.path.join(root, file), 'rb') as f:
                    tmp_data = pickle.load(f)
                    tmp_data = np.array(tmp_data[0])
                    data.append(tmp_data)

# 随机从data中拿出1000个样本‘
data = np.array(data)
print(data.shape)

scaler = StandardScaler()
embedding_normalized = scaler.fit_transform(data)

print(embedding_normalized.shape)
start_time = time.time()
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,    # 最小簇大小，根据数据调整
    min_samples=2,          # 用于估计点的密度
    cluster_selection_method='eom',  # 'eom' 或 'leaf'
)
labels = clusterer.fit_predict(embedding_normalized)
with open('bytecode_hdbscan_bge.pickle','wb') as f:
    pickle.dump(labels,f)
end_time = time.time()
print(f"聚类耗时: {end_time - start_time:.2f} 秒")

start_time = time.time()
# 可视化聚类结果
pca_2d = PCA(n_components=2)
pca_2d_result = pca_2d.fit_transform(embedding_normalized)

plt.figure(figsize=(8, 6))
plt.scatter(pca_2d_result[:,0], pca_2d_result[:, 1], c=labels, cmap='viridis', s=50)
plt.colorbar(label='Cluster Label')
plt.title('HDBSCAN Clustering Results')
plt.savefig('hdbscan_bge.png')
end_time = time.time()
print(f"PCA耗时: {end_time - start_time:.2f} 秒")


tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
data_2d = tsne.fit_transform(embedding_normalized)

# 可视化 2D t-SNE 结果
plt.figure(figsize=(20, 20))
plt.scatter(data_2d[:, 0], data_2d[:, 1], cmap='viridis')
plt.title('2D t-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('tsne_2d.png')
plt.show()