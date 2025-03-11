import tensorflow_hub as hub
import sys
import tensorflow as tf
import os
import json
import pickle
import numpy as np
from tqdm import tqdm

print(tf.__version__)
print(tf.sysconfig.get_build_info()["cuda_version"])
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

modelTA = hub.load("universal-sentence-encoder-large_5")
print(f"Loaded model success!")

def embed(input):
    try:
        result = modelTA(input)
        return result.numpy()
    except Exception:
        print("Exception in embed function\n")
        return []

def chunk_and_embed(opcodes, chunk_size=10000, overlap=1000):
    tokens = opcodes.strip().split()
    chunks = []
    
    # 如果序列较短，直接处理
    if len(tokens) <= chunk_size:
        return embed([" ".join(tokens)])[0]
        
    # 创建chunks
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) > chunk_size/2:  # 确保最后一个chunk足够长
            chunks.append(" ".join(chunk))
    
    # 获取每个chunk的嵌入并平均
    embeddings = [embed([chunk])[0] for chunk in chunks]
    return np.mean(embeddings, axis=0)

# load data
OPCODE_DIR = "/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/disasm_dataset/"
# CONTRACTS_PATH = "/data2/zuobinwang/nft-all-in-one-0114-data/rq1_demo/original_bytecode/bytecode_demo.csv"
SAVE_DIR = "/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/opcode_embeddings/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# addresses = {}
# def load_contract_addresses(json_path):
#     """Load contract address to label mapping"""
#     with open(json_path, 'r') as f:
#         for line in f:
#             address = line.split(',')[0]
#             addresses[address] = 1
            
# load_contract_addresses(CONTRACTS_PATH)

# total_contracts = len(os.listdir(OPCODE_DIR))
# print(f"Total contracts to process: {total_contracts}")
processed = 0
errors = []
results = []

for filename in tqdm(os.listdir(OPCODE_DIR),"files"):
    if filename.endswith('.disasm'):
        contract_address = filename[:-7]
        opcode_file = os.path.join(OPCODE_DIR, filename)
        
        try:
            with open(opcode_file, 'r') as f:
                opcodes = f.readlines()
            # 把opcode 开头的删掉
            opcodes = [line.split()[0] for line in opcodes]
            opcodes = [line for line in opcodes if not line.startswith('opcode')]
            opcodes = " ".join(opcodes)
            
            embedding = chunk_and_embed(opcodes)
            if len(embedding) > 0:  # 确保嵌入成功
                results.append({
                    'address': contract_address,
                    'embedding': embedding
                })
        except Exception as e:
            print(f"Error processing {contract_address}: {str(e)}")
            errors.append((contract_address, str(e)))
        
        # processed += 1
        # if processed % 50 == 0:
        #     print(f"Progress: {processed}/{total_contracts} contracts processed ({len(errors)} errors)")
# for filename in addresses.keys():
#     contract_address = filename
#     opcode_file = os.path.join(OPCODE_DIR, filename+'.disasm')
    
#     try:
#         with open(opcode_file, 'r') as f:
#             opcodes = f.read().strip()
#         embedding = chunk_and_embed(opcodes)
#         if len(embedding) > 0:  # 确保嵌入成功
#             results.append({
#                 'address': contract_address,
#                 'embedding': embedding
#             })
#     except Exception as e:
#         print(f"Error processing {contract_address}: {str(e)}")
#         errors.append((contract_address, str(e)))
    
#     processed += 1
#     if processed % 50 == 0:
#         print(f"Progress: {processed}/{total_contracts} contracts processed ({len(errors)} errors)")

with open(SAVE_DIR+'contract_embeddings_opcodes_chunk.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nProcessing complete!")
print(f"Total contracts processed: {processed}")
print(f"Total errors: {len(errors)}")