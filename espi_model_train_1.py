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

def embedding(long_text):
    nlp = spacy.load("en_core_web_sm")
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

def embedding_data(data ,batch_size):
    data_list = []
    for i in tqdm(range(len(data))):#len(data)
        text = data["msg"][i]
        if text == "":
            continue
        x, edge_index = embedding(text)
        data_list.append(Data(x = x, edge_index=edge_index, y = data["label"][i]))
    loader = DataLoader(data_list, batch_size=batch_size)
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

def run(db ,fold_num):
    torch.manual_seed(128)

    spidb = 'spidb'
    patchdb = 'patchdb'
    if db == spidb:
        traindb = spidb
        testdb = patchdb
    else:
        traindb = patchdb  
        testdb = spidb    
    train_data, test_data = load_fold_dataset("5_fold_datasets", traindb, fold_num)
    test_data_other =  load_whole_dataset("5_fold_datasets", testdb)
    batch_size = 128
    filename = traindb + '_train_'+ str(fold_num) + ',pkl'
    train_loader = load_loader(filename , train_data, batch_size)  
    filename = traindb + '_test_'+ str(fold_num) + ',pkl'
    test_loader = load_loader(filename , test_data, batch_size) 
    filename = testdb + '_test_'+ 'whole' + ',pkl'
    other_loader = load_loader(filename , test_data_other, batch_size) 

    # 为了在 GPU 上训练，将模型和数据移到 GPU 上
    model = ESPI_MSG_MODEL(hidden_size=96, layer_num=2, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in test_loader:
            data = data.to(device)  # 将数据移到 GPU 上
            optimizer.zero_grad()
            out = model(data.x, data.edge_index , data.batch)  # 前向传播
            loss = criterion(out, data.y.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
    torch.save(model, "model_"+ traindb +str(fold_num)+".pt")
    
    predicted_classes = []
    test_list = []
    for i in range(len(test_data)):
        if test_data['msg'][i] != "":
            test_list.append(test_data['label'][i])
    for data in test_loader:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        pred =  pred.to('cpu')
        res = pred.detach().numpy()
        for i in range(len(res)):
            predicted_classes.append(round(res[i]))
            
    report = classification_report(test_list, predicted_classes)   
    print(report)

    predicted_classes = []
    test_list = []
    count = 0
    for i in tqdm(range(len(test_data_other))):
        if test_data_other['msg'][i] != "":
            test_list.append(test_data_other['label'][i])
            count +=1
        else: print(i)
    print(count)
    for data in other_loader:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        pred =  pred.to('cpu')
        res = pred.detach().numpy()
        for i in range(len(res)):
            predicted_classes.append(round(res[i]))
            
            
    report = classification_report(test_list, predicted_classes)   
    print(report)

if __name__ == "__main__":
    run('spidb',0)