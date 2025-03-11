# 检查聚类结果
import csv
import json

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bytecode_cluster.json', 'r') as f:
    cluster_hash_to_clusters = json.load(f)

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bytecode_cluster_addresses.json', 'r') as f:
    cluster_addresses = json.load(f)

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/hdbscan_labeled.csv', 'r') as f:
    reader = csv.reader(f)
    labels_sourcecode = list(reader)
    labels_sourcecode = labels_sourcecode[1:]


# 观察有多少未分类的
unclassified = 0
all_contracts = 0
label_to_number_map = {}
for label in labels_sourcecode:
    cluster_hash = cluster_addresses[label[0]]
    clusters = cluster_hash_to_clusters[cluster_hash]
    if label[1] == '-1':
        unclassified += len(clusters)
    all_contracts += len(clusters)
    if label[1] not in label_to_number_map:
        label_to_number_map[label[1]] = 0
    label_to_number_map[label[1]] += len(clusters)

label_to_number_map['-1'] = 0
with open('cluster_sizes.json', 'w') as f:
    label_to_number_map = {k: v for k, v in sorted(label_to_number_map.items(), key=lambda item: item[1], reverse=True)}
    json.dump(label_to_number_map, f)
print(f"Total number of unclassified contracts: {unclassified}")
print(f"Total number of contracts: {all_contracts}")

# 有1个是未分类的
print(f"Total number of clusters: {len(set([label[1] for label in labels_sourcecode]))-1}")

# 按从大到小排序，并且列出cdf
from collections import Counter
import matplotlib.pyplot as plt

# 排除掉-1
# 现在需要统计每个聚类的大小

# cluster_sizes = Counter([label[1] for label in labels_sourcecode if label[1] != '-1'])
# cluster_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
cluster_sizes =  sorted(label_to_number_map.items(), key=lambda x: x[1], reverse=True)

# Top 30 对应的类是哪些
print("Top 10 clusters:")
for cluster_id, size in cluster_sizes[:11]:
    print(f"Cluster {cluster_id}: {size} contracts")
# Top 30 clusters:
# Cluster 470: 23425 contracts
# Cluster 1143: 22863 contracts
# Cluster 338: 8592 contracts
# Cluster 13: 1858 contracts
# Cluster 290: 1523 contracts
# Cluster 142: 1497 contracts
# Cluster 2480: 1387 contracts
# Cluster 1441: 1255 contracts
# Cluster 2645: 1055 contracts
# Cluster 129: 760 contracts
# Cluster 2501: 649 contracts
# Cluster 37: 607 contracts
# Cluster 2178: 519 contracts
# Cluster 414: 367 contracts
# Cluster 1613: 366 contracts
# Cluster 2606: 342 contracts
# Cluster 113: 305 contracts
# Cluster 600: 300 contracts
# Cluster 312: 289 contracts
# Cluster 1329: 269 contracts
# Cluster 537: 261 contracts
# Cluster 2186: 240 contracts
# Cluster 532: 234 contracts
# Cluster 217: 212 contracts
# Cluster 115: 212 contracts
# Cluster 1091: 209 contracts
# Cluster 253: 202 contracts
# Cluster 2206: 200 contracts
# Cluster 291: 198 contracts
# Cluster 2490: 195 contracts
# Cluster 921: 192 contracts
# Cluster 1468: 191 contracts
# Cluster 824: 189 contracts
# Cluster 2175: 186 contracts
# Cluster 462: 184 contracts
# Cluster 2644: 181 contracts
# Cluster 100: 178 contracts
# Cluster 2709: 178 contracts
# Cluster 319: 176 contracts
# Cluster 184: 167 contracts
# Cluster 48: 163 contracts
# Cluster 2465: 156 contracts
# Cluster 607: 154 contracts
# Cluster 1816: 153 contracts
# Cluster 266: 144 contracts
# Cluster 326: 141 contracts
# Cluster 1839: 141 contracts
# Cluster 2609: 140 contracts
# Cluster 868: 138 contracts
# Cluster 1527: 137 contracts
cdf = []
total = 0
for cluster_id, size in cluster_sizes:
    total += size
    cdf.append(total)

# 要百分比
cdf = [x/6004859*100 for x in cdf]

# 还要前30个占比多少
print(f"Top 10 clusters account for {cdf[50]:.2f}% of all contracts")
plt.plot(cdf)

# 在图中体现出来,要一个横虚线，一个竖虚线
plt.axhline(y=cdf[50], color='r', linestyle='--')
plt.axvline(x=50, color='r', linestyle='--')
plt.text(50, cdf[50], f'({50}, {cdf[50]:.2f}%)', color='red', ha='left', va='bottom')
plt.xlabel("Number of Clusters")
plt.ylabel("% of Contracts")
plt.title("CDF of Cluster Sizes")
plt.savefig('cdf.png')

import os

# 读取合约代码
def contract_has_code(address):
    # 根据实际情况设置文件路径
    filepath = f'/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/sourcecode/{address}.sol'  # 示例路径
    if os.path.exists(filepath):
        return True
    return False

total = 0
total_has_label = 0
label_map = {}
label_all = {}
# 判断每一个聚类中是否有代码
for label in labels_sourcecode:
    cluster_id = label[1]
    address = label[0]
    if cluster_id not in label_all:
        label_all[cluster_id] = 0
    label_all[cluster_id] += 1
    total += 1
    if cluster_id not in label_map:
        label_map[cluster_id] = 0
    if contract_has_code(address):
        total_has_label += 1
        label_map[cluster_id] += 1

print(f"Total number of contracts: {total}")
print(f"Total number of contracts with code: {total_has_label}")
print('label_map[-1]:', label_map['-1'])
print('label_all[-1]:', label_all['-1'])
print('else_label:', total_has_label - label_map['-1'])
print('else_all:', total - label_all['-1'])
# 计算每个聚类中有代码的比例
label_ratio = {cluster_id: label_map[cluster_id] / label_all[cluster_id] for cluster_id in label_map}

# 判断top 50是否都有源代码
top_clusters = [cluster_id for cluster_id, size in cluster_sizes[:50]]
top_clusters_has_code = [cluster_id for cluster_id in top_clusters if label_ratio[cluster_id] > 0]
print(f"Top 50 clusters with code: {len(top_clusters_has_code)}")
# 没有代码的聚类数量
no_code_clusters = len([cluster_id for cluster_id, ratio in label_ratio.items() if ratio == 0])
print(f"Total number of clusters without code: {no_code_clusters}")

plt.figure()
# 含有代码的比例分布图
plt.hist(label_ratio.values(), bins=50)
plt.xlabel("Code Ratio")
plt.ylabel("Number of Clusters")
plt.title("Distribution of Code Ratios")
plt.savefig('code_ratio.png')


# 对聚类进行可视化，使用t-SNE
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import pickle
# 读取聚类结果
with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/hdbscan_labeled.csv', 'r') as f:
    reader = csv.reader(f)
    labels_sourcecode = list(reader)
    labels_sourcecode = labels_sourcecode[1:]
    
# 首先需要读取Top 50 的聚类
top_clusters = [cluster_id for cluster_id, size in cluster_sizes[:10]]
# 读取这些聚类对应的合约地址
top_cluster_addresses = []
for label in labels_sourcecode:
    if label[1] in top_clusters:
        top_cluster_addresses.append(label[0])

print(f"Total number of contracts in top clusters: {len(top_cluster_addresses)}")

# 获取这些地址对应的聚类标签和向量
vector_data = []
cluster_labels = []
for root, dirs, files in os.walk('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bge_embedding_output/'):
    for file in tqdm(top_cluster_addresses, 'Reading files'):
        with open(os.path.join(root, file + ".pickle"), 'rb') as f:
            tmp_data = pickle.load(f)
            tmp_data = np.array(tmp_data[0])  # Assuming the first item in the list is the embedding vector
            vector_data.append(tmp_data)
            
            # 找到这个地址对应的聚类标签
            for label in labels_sourcecode:
                if label[0] == file:  # 地址对应上文件名
                    cluster_labels.append(label[1])

# 转换为numpy数组
vector_data = np.array(vector_data)

# 使用t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(vector_data)

# 可视化
plt.figure(figsize=(10, 8))

# 不同聚类标签用不同颜色区分
unique_labels = set(cluster_labels)
for label in unique_labels:
    indices = [i for i, l in enumerate(cluster_labels) if l == label]
    alphas = [0.8 if contract_has_code(top_cluster_addresses[i]) else 0.3 for i in indices]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f"Cluster {label}", alpha=alphas)
    

plt.title('t-SNE Visualization of Clustered Contracts')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.savefig('tsne_sourcecode.png')

