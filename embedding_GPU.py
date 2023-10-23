import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import global_mean_pool
import spacy
import numpy as np
from load_dataset import load_fold_dataset, load_whole_dataset
from tqdm import tqdm

long_text = ""

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")
#device = torch.device("cpu")
# 修改嵌入层和GRU的维度
class GGNN(nn.Module):
    def __init__(self, num_features, output_size):
        super(GGNN, self).__init__()
        self.embedding_size = 128
        self.conv1 = GraphConvolution(num_features, self.embedding_size)
        self.output_size = output_size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, data.batch)  # 使用全局平均池化
        return x

class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index):
        return self.lin(x_j)
    
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
    return x,edge_index

# 创建一个简单的图数据集
class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['custom_data.pt']

    def download(self):
        pass

    def process(self ):
        x,edge_index=embedding(long_text)
        x, edge_index = x.to(device), edge_index.to(device)
        data = Data(x=x, edge_index=edge_index)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def ensembling(graph_vector):
    fc = nn.Linear(128, 1)
    # 计算 prob
    prob = torch.sigmoid(fc(graph_vector))
    # 根据 prob 得到 prediction
    prediction = 1 if prob >= 0.5 else 0
    return prob,prediction

if __name__ == "__main__" :  
    torch.manual_seed(0)
    #msg_data = pd.read_csv("qume_part.csv")
    train_data, test_data = load_fold_dataset("5_fold_datasets", "spidb", 0)
    #msg_data = pd.read_csv("qemu.csv")
    train_vector_list = []
    test_vector_list = []
    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # 初始化和使用数据集
    for i in tqdm(range(len(train_data)), desc="Processing"):
        long_text =  train_data['msg'][i]
        if long_text == "":
            train_vector_list.append(np.zeros(128))
            continue
        dataset = CustomDataset(root='custom', transform=NormalizeFeatures())
        dataset.process()
        #dataset = Planetoid(root='cora', name='Cora', transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        
        # 初始化模型
        model = GGNN(num_features=dataset.num_features, output_size=128).to(device)  # 修改输出大小为128
        
        # 获得图的向量表示
        graph_vector = model(data)
        train_vector_list.append(graph_vector[0].tolist())

    traindata_array = np.array(train_vector_list)
    np.save('train_vector.npy', traindata_array)

    for i in tqdm(range(len(test_data)), desc="Processing"):
        long_text =  test_data['msg'][i]
        if long_text == "":
            test_vector_list.append(np.zeros(128))
            continue
        dataset = CustomDataset(root='custom', transform=NormalizeFeatures())
        dataset.process()
        data = dataset[0].to(device)
        
        # 初始化模型
        model = GGNN(num_features=dataset.num_features, output_size=128).to(device)  # 修改输出大小为128
        
        # 获得图的向量表示
        graph_vector = model(data)
        test_vector_list.append(graph_vector[0].tolist())
    testdata_array = np.array(test_vector_list)
    np.save('test_vector.npy', testdata_array)
        #print(graph_vector)
    #     prob,prediction = ensembling(graph_vector)
    #     print("Prob:", prob.data[0][0].to(device))
    #     print("Prediction:", prediction)
    #     if(prediction == msg_data['vulnerability'][i]):
    #         acc = acc + 1
    #     if(prediction == 1 and msg_data['vulnerability'][i] == 1):
    #         tp = tp + 1
    #     if(prediction == 1 and msg_data['vulnerability'][i] == 0):
    #         fp = fp + 1
    #     if(prediction == 0 and msg_data['vulnerability'][i] == 0):
    #         tn = tn + 1
    #     if(prediction == 0 and msg_data['vulnerability'][i] == 1):
    #         fn = fn + 1

    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)
    # f1 = (2 * precision * recall) / ( precision + recall)
    # print("acc_rate:",acc/len(msg_data))
    # print("recall:",recall)
    # print("precision:",precision)
    # print("F1-score:",f1)


    #np.save('vector_list.npy', data_array)
    #print(vector_list)