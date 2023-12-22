import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset,DataLoader
import os
from torch_geometric.data import Dataset, DataLoader, Data
from torch_geometric.transforms import NormalizeFeatures
import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.metrics import classification_report
import dill
from load_dataset import load_fold_dataset,load_whole_dataset
from espi_model import ESPI_MSG_MODEL
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
import json

def embedding(long_text,nlp):
    
    # 创建节点特征 x
    doc = nlp(long_text)
    lines = long_text.splitlines()
    x = torch.zeros(len(doc), 96)

    i = 0
    for token in doc:
        word = token.text
        word_vec = nlp(word)
        # 获取单词的96维嵌入向量
        x[i] = torch.tensor(word_vec.vector)
        i = i + 1

    # 创建边的索引 edge_index
    edge_list = []
    node_counter = 0
    for line in lines:
        doc = nlp(line)
        for token in doc:
            if token.head is not token:
                edge_list.append([token.i + node_counter, token.head.i + node_counter])
        node_counter += len(doc)
    edge_index = torch.tensor(edge_list).t()
    return x, edge_index

def process_row(row):
    index, data = row
    text = data["msg"]
    if pd.isnull(text) or re.match(r'^[ \n]*$', text) or text == '': 
        x = torch.zeros(1, 96)
        edge_index = torch.zeros(2, 1, dtype=torch.int)
    else:
        nlp = spacy.load("en_core_web_sm")
        x, edge_index = embedding(text, nlp)
    return Data(x=x, edge_index=edge_index, y=data["label"])

def embedding_task_parallel(data):
    data_list_whole = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_row, data.iterrows(), chunksize=32), total=len(data)))

    for result in results:
        data_list_whole.append(result)
    return data_list_whole

def embedding_task(data):
    data_list_whole = []
    nlp = spacy.load("en_core_web_sm")
    for i in tqdm(range(len(data))):
        text = data["msg"][i]
        text = re.sub(r'^\n+$', '', text)
        if  pd.isnull(text) or re.match(r'^[ \n]*$', text) or text == '': 
            x = torch.zeros(1, 96)
            edge_index = torch.zeros(2, 1, dtype=torch.int)
        else:
            x, edge_index = embedding(text,nlp)
        data_list_whole.append(Data(x = x, edge_index=edge_index, y = data["label"][i]))
    return data_list_whole

def embedding_data(data ,batch_size ):
    data_list_whole = []
    nlp = spacy.load("en_core_web_sm")
    if len(data) < 30000:
        data_list_whole = embedding_task_parallel(data)
    else:
        data_part = data.loc[:29999]
        data_list_whole = embedding_task_parallel(data_part)
        for i in tqdm(range(30000,len(data))):
            text = data["msg"][i]
            if pd.isnull(text) or re.match(r'^[ \n]*$', text) or text == '' : 
                x = torch.zeros(1, 96)
                edge_index = torch.zeros(2, 1, dtype=torch.int)
            else:
                x, edge_index = embedding(text,nlp)
            data_list_whole.append(Data(x = x, edge_index=edge_index, y = data["label"][i]))
    loader = DataLoader(data_list_whole , batch_size=batch_size)

    return loader

def load_loader(filename , data ,batch_size):
    if (os.path.exists(filename)):
        with open(filename,'rb') as f:
            loader = dill.load(f)        
    else:
        loader = embedding_data(data ,batch_size)
        with open(filename ,'wb') as f:   
            dill.dump(loader, f) 
    return loader

def run(name):
    torch.manual_seed(128) 
    with open('ysk_dataset/'+name+'.json', 'r') as file:
        data = json.load(file)
    data = pd.DataFrame(data['data'])
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    print(data)
    print(len(data))
    batch_size = 128
    filename = name + '.pkl'
    test_loader = load_loader(filename , data, batch_size) 

def embed():
    data = pd.read_excel('test_data/test_filter.xlsx', index_col=False)
    print(data)
    print(len(data))
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    batch_size = 128
    filename = 'test_filter.pkl'
    test_loader = load_loader(filename , data, batch_size) 


if __name__ == "__main__":
    run('sp-test(1)')
    #run('sp-train(1)')
    # run('spidb_fold')
    # run('patchdb_fold')
    #embed()
        
        
        