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
import numpy as np
import re
from multiprocessing import Pool
from itertools import chain
from multiprocessing import Process, Manager
import json
from concurrent.futures import ProcessPoolExecutor

def embedding(long_text):
    nlp = spacy.load("en_core_web_sm")
    # 创建节点特征 x
    doc = nlp(long_text)
    #lines = re.split(r'[.\n]', long_text)
    #lines = [lines.strip() for lines in lines if lines.strip()]
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

def embedding_data_worker(args):
    i, text, label = args

    if re.match(r'^[ \n]*$', text):
        x = torch.zeros(0, 96)
        edge_index = torch.zeros(2, 0, dtype=torch.int)
        return Data(x=x, edge_index=edge_index, y=label)
    x, edge_index = embedding(text)
    return Data(x=x, edge_index=edge_index, y=label)

def parallel_pool(chunk_args,num_workers,iter):
    if iter < 30000:
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(embedding_data_worker, chunk_args, chunksize=32), total=len(chunk_args)))
        #with Pool(num_workers) as pool:
        #    chunk_data = list(tqdm(pool.imap(embedding_data_worker, chunk_args),total=len(chunk_args)))
    else:
        chunk_data =[]
        for i in tqdm(range(len(chunk_args))):
            chunk_data.append(embedding_data_worker(chunk_args[i]))
            
    return chunk_data

def embedding_data_parallel(data, batch_size, num_workers=24, chunk_size=10000):
    msg = data['msg']
    labels = data['label']
    args_list = [(i, text, labels[i]) for i, text in enumerate(msg)]
    data_list = []
    for i in range(0, len(args_list), chunk_size):
        chunk_args = args_list[i:i+chunk_size]
        chunk_data = parallel_pool(chunk_args,num_workers,i)
        data_list.extend(chunk_data)

    loader = DataLoader(data_list, batch_size=batch_size)
    return loader

def load_loader(filename , data ,batch_size):
    if (os.path.exists(filename)):
        with open(filename,'rb') as f:
            loader = dill.load(f)        
    else:

        loader = embedding_data_parallel(data, batch_size)
        with open(filename ,'wb') as f:   
            dill.dump(loader, f) 
    return loader

def train(name, train_data, num_epochs, batch_size, model_output_path):
    torch.manual_seed(128)
    print(len(train_data))
    filename = name + '_batch'+ str(batch_size) + '.pkl'
    filename = 'spidb_test_whole.pkl'
    train_loader = load_loader(filename , train_data, batch_size)  
    model = ESPI_MSG_MODEL(hidden_size=96, layer_num=2, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = 999999.9

    loss_times = 0 
    # 训练循环
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)  # 将数据移到 GPU 上
            optimizer.zero_grad()
            out = model(data.x, data.edge_index , data.batch)  # 前向传播
            if type(data.y).__name__ == 'list':
                target_tensor = torch.tensor(data.y).float().to(device)  # 将其移动到与模型输出相同的设备
                loss = criterion(out, target_tensor)
            else:
                loss = criterion(out, data.y.float())  # 计算损失
            #loss = criterion(out, data.y.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            loss_times = 0
        else:
            loss_times += 1
        if loss_times >= 10:
            print("early stop : no improvements for 10 epochs")
            break
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    torch.save(model, model_output_path)
    return model

def test(name,test_data, model, batch_size, output_path):
    torch.manual_seed(128)
    print(len(test_data))
    filename = name + '_batch'+ str(batch_size) + '.pkl'
    filename = 'sp-test(1).pkl'
    test_loader = load_loader(filename , test_data, batch_size)  
    predicted_classes = []
    label = test_data['label']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in test_loader:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        pred =  pred.to('cpu')
        res = pred.detach().numpy()
        for i in range(len(res)):
            predicted_classes.append(round(res[i]))
    report = classification_report(label, predicted_classes, output_dict=False)
    df = pd.DataFrame(test_data)
    df['predict'] = predicted_classes
    df.to_excel(output_path,index=False)   
    return report

if __name__ == "__main__": 
    #test_data =  load_whole_dataset("5_fold_datasets", "spidb")
    train_data =  load_whole_dataset("5_fold_datasets", "spidb")
    #train_data = pd.read_csv("spidb.csv",index_col=False)
    #test_data = pd.read_csv("patchdb.csv",index_col=False
    with open('ysk_dataset/sp-test(1).json', 'r') as file:
        data = json.load(file)
    data = pd.DataFrame(data['data'])
    test_data = data
    #train_data = pd.read_csv("patchdb.csv",index_col=False)
    #test_data = pd.read_csv("padb.csv",index_col=False)
    model = train(name="patchdb", train_data=train_data, num_epochs=30, batch_size=128, model_output_path="espi_output/model_spi.pt")
    #test_data = pd.read_excel("testset.xlsx",index_col=False)
    #model = torch.load("model_patchdb2.pt")
    report = test(name="sp-test(1)",test_data=test_data, model=model, batch_size=128, output_path="espi_output/train_sp-train_test_patch_result.xlsx")
    print(report)

#name：为这个嵌入的数据起一个名字，名字相同的不会重复进行词嵌入
#输入的train/test_data可以是各种类型，Dataframe和json都行        
#修改embedding_data_parallel函数可调整跑的CPU核数，默认16核，总共36核