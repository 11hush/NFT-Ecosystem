from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm
import json

data = []

# total = 0
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

with open('bytecode_hdbscan_bge.pickle','rb') as f:
    labels = pickle.load(f)
    
filtered_data = embedding_normalized[labels != -1]
filtered_labels = labels[labels != -1]
def noise_ratio(labels):
    noise_points = (labels == -1).sum()
    print(f"Noise Points: {noise_points}")
    total_points = len(labels)
    return noise_points / total_points

noise = noise_ratio(labels)
print(f"Noise Ratio: {noise:.2%}")
score = silhouette_score(filtered_data, filtered_labels)
print(f"Silhouette Score: {score}")

from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(filtered_data, filtered_labels)
print(f"Davies-Bouldin Index: {dbi}")


