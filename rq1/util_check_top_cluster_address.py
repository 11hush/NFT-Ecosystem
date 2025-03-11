import os
import json
import csv

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bytecode_cluster.json', 'r') as f:
    cluster_hash_to_clusters = json.load(f)

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bytecode_cluster_addresses.json', 'r') as f:
    cluster_addresses = json.load(f)

with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/hdbscan_labeled.csv', 'r') as f:
    reader = csv.reader(f)
    labels_sourcecode = list(reader)
    labels_sourcecode = labels_sourcecode[1:]
    

# 按从大到小排序，并且列出cdf
from collections import Counter
import matplotlib.pyplot as plt

# 排除掉-1
cluster_sizes = Counter([label[1] for label in labels_sourcecode if label[1] != '-1'])
cluster_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

# top_cluster = [cluster_id for cluster_id, size in cluster_sizes[:10]]
top_cluster = ['13','342','470','1143','133','267','338','292','1633','14','155']
print('top_cluster:', top_cluster)

# 读取这些聚类对应的合约地址
top_cluster_addresses_origin = {
}
for x in top_cluster:
    top_cluster_addresses_origin[x] = []

for label in labels_sourcecode:
    if label[1] in top_cluster and len(top_cluster_addresses_origin[label[1]]) <20:
        top_cluster_addresses_origin[label[1]].append(label[0])
        

with open('./top_10_cluster_addresses.json', 'w') as f:
    json.dump(top_cluster_addresses_origin, f)
