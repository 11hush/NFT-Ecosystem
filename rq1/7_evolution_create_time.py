
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/top_cluster_addresses.json', 'r') as f:
    top_cluster_addresses = json.load(f)
cluster_times = []
node_name = []

top_clusters = ['470', '1143', '338', '13', '290', '142', '2480', '1441', '2645', '129']
# top_clusters = ['13','342','470','1143','133','267','338','292','1633','14','155']

node_create_time = {}
with open('./nodes_with_create_time.json', 'r') as f:
    node_create_time = json.load(f)

cluster_times = {}
for cluster in top_clusters:
    cluster_times[cluster] = []
for node in top_cluster_addresses:
    if top_cluster_addresses[node] not in top_clusters:
        continue
    if node in node_create_time:
        create_time = node_create_time[node]
        cluster_times[top_cluster_addresses[node]].append(create_time)

with open('cluster_times.json', 'w') as f:
    json.dump(cluster_times, f)

import matplotlib.pyplot as plt
import pandas as pd


# 起始时间戳，2015年7月31日
start_timestamp = 1438268400
max_timestamp = 1717281407

# 将时间戳转换为 datetime 格式
start_time = pd.to_datetime(start_timestamp, unit='s')
max_time = pd.to_datetime(max_timestamp, unit='s')

# 处理时间区间并统计每个区间内的数量
def get_time_intervals(times, start_timestamp):
    times = pd.to_datetime(times, unit='s')
    period_index = pd.date_range(start=start_time, end=max_time, freq='3M')
    intervals = pd.cut(times, bins=period_index, right=False)
    interval_counts = intervals.value_counts().sort_index()
    interval_start_times = [interval.left for interval in interval_counts.index]
    return interval_start_times, interval_counts.values

# 准备热图数据
heatmap_data = []
cluster_labels = []
time_labels = None

for cluster_id, times in cluster_times.items():
    interval_start_times, interval_counts = get_time_intervals(times, start_timestamp)
    heatmap_data.append(interval_counts)
    cluster_labels.append(f'Cluster {cluster_id}')
    if time_labels is None:  # 只需从第一个cluster获取时间标签
        time_labels = [t.strftime('%Y-%m') for t in interval_start_times]

# 转换为numpy数组并处理空值（如果某区间没有数据，填0）
heatmap_data = np.array(heatmap_data)
if heatmap_data.shape[1] < len(time_labels):  # 如果某些时间段没数据，补0
    padding = len(time_labels) - heatmap_data.shape[1]
    heatmap_data = np.pad(heatmap_data, ((0, 0), (0, padding)), mode='constant', constant_values=0)

with open('heatmap_data.csv', 'w') as f:
    f.write(','.join(['cluster_id']+time_labels) + '\n')
    for i in range(len(heatmap_data)):
        f.write(','.join([cluster_labels[i]]+ [str(x) for x in heatmap_data[i]]) + '\n')
# 设置绘图大小
plt.figure(figsize=(20, 10))

# 绘制热图
sns.heatmap(
    heatmap_data,
    xticklabels=time_labels,
    yticklabels=cluster_labels,
    cmap="YlOrRd",  # 颜色从浅黄到深红
    norm='log',  # 使用对数刻度处理数量级差异
    cbar_kws={'label': 'Number of Deployments (log scale)'},  # 颜色条标签
)

# 设置图表标题和标签
plt.title('Contract Deployment Frequency Heatmap (3-Month Intervals)')
plt.xlabel('Time')
plt.ylabel('Cluster')
plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠
plt.tight_layout()

# 保存图表为文件
plt.savefig('cluster_evolution_heatmap.png')




# 设置绘图大小
plt.figure(figsize=(20, 10))

# 为每个 cluster 绘制曲线
for cluster_id, times in cluster_times.items():
    interval_start_times, interval_counts = get_time_intervals(times, start_timestamp)
    plt.plot(interval_start_times, interval_counts, marker='o', label=f'Cluster {cluster_id}')

# 设置图表标题和标签
plt.title('Contract Deployment Frequency in 3-Month Intervals for Each Cluster')
plt.xlabel('Time')
plt.ylabel('Number of Deployments')
plt.yscale('log')  # 使用对数刻度
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠
plt.tight_layout()

# 保存图表为文件
plt.savefig('cluster_evolution.png')


