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
    # 'dev','tokenid','token','param','owner','contracts','set','map','key','index','value','tokens','data','sol','solidity','zero','target','abi','import','using','emits','account','https','pragma','length','indexed','id','identifier','mit',
]

# 结合英文停用词和 Solidity 保留关键字
combined_stop_words = list(set(ENGLISH_STOP_WORDS).union(solidity_keywords))

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def generate_document_term_matrix(cluster_texts_set):
    """
    Generate a large document-term matrix (DTM) where each row represents a cluster
    and each column represents the frequency of a word in that cluster's text.
    
    Args:
        cluster_texts_set: Dictionary with cluster_id as key and text (string) as value.
    
    Returns:
        Document-term matrix (DTM) as numpy array, and the corresponding feature names (words).
    """
    # Initialize a CountVectorizer (without TF-IDF) to generate the term-document matrix (TDM)
    vectorizer = CountVectorizer(
        stop_words=combined_stop_words,   # Remove stop words like common words
        max_df=0.999,  # Remove terms that appear in more than 85% of the documents
        min_df=1,     # Remove terms that appear in fewer than 2 documents
        ngram_range=(1, 1),  # Only use single words (unigrams)
        max_features=100000   # Limit the vocabulary size (optional)
    )
    
    # Combine all cluster texts into a single list, where each element is the text for one cluster
    all_texts = list(cluster_texts_set.values())
    
    # Fit the vectorizer to the combined list of cluster texts
    X = vectorizer.fit_transform(all_texts)
    
    # Get the feature names (words in the vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the term-document matrix (X) to a dense matrix
    dtm = X.toarray()  # n x m matrix (n = number of clusters, m = number of words)
    
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
        representative_words[i] = [word for word, _ in word_weights[:20]]
    
    return representative_words


# Example usage
cluster_texts_set = {
    1: "I love machine learning, Machine learning is fun, I enjoy coding with Python",
    2: "Blockchain technology is amazing, Ethereum is a decentralized platform, Smart contracts are useful"
}

# Generate the document-term matrix for all clusters
representative_words = generate_document_term_matrix(cluster_texts_set)

# Print the resulting Document-Term Matrix (DTM)
print("Document-Term Matrix (DTM):")
for cluster_id, words in representative_words.items():
    print(f"Cluster {cluster_id} Representative Words: {', '.join(words)}")