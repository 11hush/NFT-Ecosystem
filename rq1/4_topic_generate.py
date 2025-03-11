import pickle
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import csv
import json


# 加载聚类结果
with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/hdbscan_labeled.csv', 'r') as f:
    labels_sourcecode = []
    reader = csv.DictReader(f)
    for row in reader:
        labels_sourcecode.append({
            'address': row['ca'],
            'cluster': int(row['cluster'])
        })
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Solidity 保留关键字
solidity_keywords = [
    'address', 'bool', 'string', 'int', 'uint', 'uint256', 'uint8', 'uint16', 'uint32',
    'uint64', 'uint128', 'uint224', 'uint240', 'uint248', 'int8', 'int16', 'int32',
    'int64', 'int128', 'int224', 'int240', 'int248', 'byte', 'bytes', 'bytes1', 'bytes2',
    'bytes4', 'bytes8', 'bytes16', 'bytes32', 'mapping', 'struct', 'contract', 'function',
    'event', 'modifier', 'enum', 'public', 'private', 'internal', 'external', 'pure',
    'view', 'payable', 'nonpayable', 'returns', 'memory', 'storage', 'calldata', 'require',
    'assert', 'revert', 'if', 'else', 'while', 'for', 'do', 'break', 'continue', 'return',
    'emit', 'msg', 'tx', 'block', 'this', 'super', 'new', 'delete', 'true', 'false',
    'constructor', 'fallback', 'receive', 'virtual', 'override', 'constant', 'immutable',
    'abstract', 'try', 'catch', 'assembly',
    'dev','tokenid','token','param','owner','contracts','set','map','key','index','value','tokens','data','sol','solidity','zero','target','abi','import','using','emits','account','https','pragma','length','indexed','id','identifier','mit',
]

# 结合英文停用词和 Solidity 保留关键字
combined_stop_words = list(set(ENGLISH_STOP_WORDS).union(solidity_keywords))

# 提取聚类中的合约地址
def group_by_cluster(labels):
    clusters = {}
    for item in labels:
        cluster_id = item['cluster']
        address = item['address']
        clusters.setdefault(cluster_id, []).append(address)
    return clusters

# 读取合约代码
def read_contract_code(address):
    # 根据实际情况设置文件路径
    filepath = f'/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/sourcecode/{address}.sol'  # 示例路径
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

# 提取合约的标识符和注释
def extract_text_from_code(code):
    identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code)
    comments = re.findall(r'//.*?$|/\*.*?\*/', code, re.DOTALL | re.MULTILINE)
    return ' '.join(identifiers)


def generate_cluster_topics(cluster_texts_set):
    """
    Generate a large document-term matrix (DTM) where each row represents a cluster
    and each column represents the frequency of a word in that cluster's text.
    
    Args:
        cluster_texts_set: Dictionary with cluster_id as key and text (string) as value.
    
    Returns:
        Document-term matrix (DTM) as numpy array, and the corresponding feature names (words).
    """
    # Initialize a CountVectorizer (without TF-IDF) to generate the term-document matrix (TDM)
    print("Generating document-term matrix...")
    vectorizer = CountVectorizer(
        stop_words=combined_stop_words,   # Remove stop words like common words
        max_df=0.999,  # Remove terms that appear in more than 85% of the documents
        min_df=1,     # Remove terms that appear in fewer than 2 documents
        ngram_range=(1, 1),  # Only use single words (unigrams)
        max_features=100000   # Limit the vocabulary size (optional)
    )
    
    # Combine all cluster texts into a single list, where each element is the text for one cluster
    all_texts = list(cluster_texts_set.values())
    
    cluster_ids = list(cluster_texts_set.keys())
    
    # Fit the vectorizer to the combined list of cluster texts
    X = vectorizer.fit_transform(all_texts)
    
    # Get the feature names (words in the vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the term-document matrix (X) to a dense matrix
    dtm = X.toarray()  # n x m matrix (n = number of clusters, m = number of words)
    
    print("Document-term matrix generated.")
    # Calculate total word counts per document (cluster)
    total_counts = np.sum(dtm, axis=1)
    
    # Calculate total word counts for all documents (to compute the overall frequency of words)
    total_counts_all = np.sum(dtm)
    
    # Calculate the overall frequency of each word across all documents
    word_frequencies = np.sum(dtm, axis=0) / total_counts_all  # Sum over all clusters, normalize
    
    # Initialize the dictionary to store representative words for each cluster
    representative_words = {}

    # Iterate through each document (cluster)
    for i, row in enumerate(dtm):
        cluster_id = cluster_ids[i]
        print(f"Processing cluster {cluster_id}...")
        # Calculate the frequency of words in the current cluster (document)
        cluster_total = total_counts[i]
        
        # Calculate word weights for the current cluster (for each word in the vocabulary)
        word_weights = []
        
        for j, word_count in enumerate(row):
            # Calculate word frequency in the current cluster (document)
            word_cluster_freq = word_count / cluster_total
            
            # Calculate the word weight (frequency in this document minus global frequency)
            word_weight = word_cluster_freq - word_frequencies[j]
            
            # Store word and its calculated weight
            word_weights.append((feature_names[j], word_weight))
        
        # Sort words by weight (higher weight means more representative)
        word_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Select top_n representative words for this cluster
        representative_words[cluster_id] = [word for word, _ in word_weights[:20]]
    
    return representative_words

# 主程序
if __name__ == "__main__":
    # 分组聚类中的合约
    clusters = group_by_cluster(labels_sourcecode)

    cluster_texts_set = {}
    # 针对每个聚类生成主题
    tt = 0
    for cluster_id, addresses in clusters.items():
        if cluster_id == -1:
            continue
        print(f"Cluster {cluster_id}:")
        
        # 合并该聚类所有合约的代码
        cluster_texts = ""
        t = 0
        for address in addresses:
            code = read_contract_code(address)
            if code:
                cluster_texts+=extract_text_from_code(code)
            t += 1
            # if t > 100:
            #     break
        
        # 如果该聚类没有合约代码，跳过
        if not cluster_texts:
            print("  No contracts found for this cluster.")
            continue
        cluster_texts_set[cluster_id] = cluster_texts
        
        tt += 1
        # if tt > 100:
        #     break
        # # 生成主题
        # topic_keywords = generate_topic(cluster_texts, n_topics=1)
        # print("  Topic Keywords:", ", ".join(topic_keywords))
    # Use the function
    cluster_topics = generate_cluster_topics(cluster_texts_set)

    result = []
    for cluster_id, topics in cluster_topics.items():
        result.append({
            'cluster_id': cluster_id,
            'topics': topics
        })
    with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/cluster_topics.pkl', 'wb') as f:
        pickle.dump(result, f)
    with open('/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/cluster_topics.json', 'w') as f:
        json.dump(result, f)
    print(f"\nProcessing complete!")
    for cluster_id, topics in cluster_topics.items():
        print(f"Cluster {cluster_id}: {topics}")