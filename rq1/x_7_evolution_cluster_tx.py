
import os
import json
# 遍历nodes文件夹，获取所有的json文件
with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/top_cluster_addresses.json', 'r') as f:
    top_cluster_addresses = json.load(f)
cluster_times = []
node_name = []

# top_clusters = ['470', '1143', '338', '13', '290', '142', '2480', '1441', '2645', '129', '2501', '37', '2178', '414', '1613', '2606', '113', '600', '312', '1329', '537', '2186', '532', '217', '115', '1091', '253', '2206', '291', '2490', '921', '1468', '824', '2175', '462', '2644', '100', '2709', '319', '184', '48', '2465', '607', '1816', '266', '326', '1839', '2609', '868', '1527']
top_clusters = ['470', '1143', '338', '13', '290', '142', '2480', '1441', '2645', '129']
# 获取所有节点文件
node_files = []
for root, dirs, files in os.walk('./nodes'):
    for file in files:
        if file.endswith('.json'):
            node_files.append(os.path.join(root, file))

# 按照文件名中的数字顺序排序
node_files.sort(key=lambda x: int(x.split('_')[1].split('to')[0]))

# 处理文件
for file_path in node_files:
    file_name = os.path.basename(file_path)
    node_name.append(int(file_name.split('_')[1].split('to')[0]))
    current_cluster_times = {}
    for cluster in top_clusters:
        current_cluster_times[cluster] = 0
    with open(file_path, 'r') as f:
        data = json.load(f)
        for d in data:
            if d in top_cluster_addresses and top_cluster_addresses[d] in top_clusters:
                # if top_cluster_addresses[d] not in current_cluster_times:
                #     current_cluster_times[top_cluster_addresses[d]] = 0
                current_cluster_times[top_cluster_addresses[d]] += 1
    cluster_times.append(current_cluster_times)
    print(file_name, current_cluster_times)
print('node_name:', node_name)
print('cluster_times:', cluster_times)

import matplotlib.pyplot as plt

# 绘制图表
# 每个聚类是如何随时间变化的

plt.figure(figsize=(20, 10))

for cluster in top_clusters:
    y = [times[cluster] for times in cluster_times]
    # y轴 log scale
    plt.yscale('log')
    plt.plot(node_name, y, label=f'Cluster {cluster}')
plt.xlabel('Time')
plt.ylabel('Number of Contracts')
plt.legend()
plt.savefig('cluster_evolution.png')
