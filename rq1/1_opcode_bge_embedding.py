
import os
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import random
import logging
import multiprocessing

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pickle

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('Eembedding')


class TextEmbedding():
    def __init__(self):
        self.tokenizer = None
        self.model = None
    
    def get_embedding(self, input_texts: list, batch_size: int = 32) -> torch.Tensor:        
        # Function to process a single batch and get embeddings
        def process_batch(batch_texts):
            batch_dict = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                # for k, v in batch_dict.items():
                #     logger.info(f"Debug tensor {k} shape: {v.shape}")
                outputs = self.model(**batch_dict)
                embeddings = outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        # Process all input texts in mini-batches
        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            embeddings = process_batch(batch_texts)
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings into a single tensor
        all_embeddings = torch.cat(all_embeddings, dim=0).to('cpu')
        
        logger.debug(f"All embeddings device: {all_embeddings.device}")
        logger.debug(f"All embeddings shape: {all_embeddings.shape}")
        logger.debug(f"Sentence embeddings: {all_embeddings}")
        
        return all_embeddings.tolist()


class BgeEmbedding(TextEmbedding):
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained('./bge-large-en-v1.5-tokenizer')
        self.model = AutoModel.from_pretrained('./bge-large-en-v1.5-model')
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"BgeEmbedding initialized in {self.device}")
    
    def __str__(self):
        return "BgeEmbedding"



def get_tokenizer_data(input_texts):
    model = BgeEmbedding()
    batch_dict = model.tokenizer(input_texts, padding=True, return_tensors='pt')
    input_ids = batch_dict['input_ids']
    # [  240,   281,    33, 15898,    23,    23,  2028]
    # [240, 281,  33, 512,  23,  23, 512]
    token_counts = (input_ids != model.tokenizer.pad_token_id).sum(dim=1)
    return token_counts.tolist()


def process(input_dir, output_dir, model):
    # 遍历input_dir下的所有文件
    dataset = []
    for root, dirs, files in os.walk(input_dir):
        for file in  tqdm(files,"wow"):
            if file.endswith(".disasm"):
                output_file = os.path.join(output_dir, file.replace('.disasm', '.pickle'))
                if os.path.exists(output_file):
                    logger.info(f"Skip existing file: {output_file}")
                    continue
                with open(os.path.join(root, file), 'r') as f:
                    nodes_text = f.readlines()
                    nodes_text = ' '.join(nodes_text)
                    nodes_text = [nodes_text]
                embedding = model.get_embedding(nodes_text)
                output_file = os.path.join(output_dir, file.replace('.disasm', '.pickle'))
                with open(output_file, 'wb') as f:
                    pickle.dump(embedding, f)
    





def embedding_inference(input_dir, output_dir, debug=False):
    # init models
    models = []
    for i in [5]:
        models.append(BgeEmbedding(device=i))
    
    # read data
    process(input_dir, output_dir, models[0])



if __name__ == '__main__':
    # nodes_text = get_nodes_text("data/mylyn/60/seed_expanded/1_step_seeds_0_expanded_model.xml")
    # model = BgeEmbedding()
    # model.get_embedding(nodes_text)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/disasm_dataset/', help='input directory')
    parser.add_argument('--output_dir', type=str, default='/data2/zuobinwang/nft-all-in-one-0114-data/rq1_all/bge_embedding_output/', help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    embedding_inference(args.input_dir, args.output_dir, args.debug)


