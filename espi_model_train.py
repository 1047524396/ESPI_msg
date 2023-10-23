import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import NormalizeFeatures
import pandas as pd
import spacy

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
        return ["custom_data.pt"]

    def download(self):
        pass

    def process(self):
        x, edge_index = embedding(long_text)
        data = Data(x=x, edge_index=edge_index)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


torch.manual_seed(128)
msg_data = pd.read_csv("qemu.csv", index_col=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESPI_MSG_MODEL(hidden_size=96, layer_num=2, dropout=0.1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

for i in range(len(msg_data)):
    long_text = msg_data["commit_msg"][i]
    dataset = CustomDataset(root="custom", transform=NormalizeFeatures())
    dataset.process()
    data = dataset[0].to(device)

    optimizer.zero_grad()

    pred = model(data.x, data.edge_index)
    gt = torch.tensor([msg_data["vulnerability"][i]], dtype=torch.float32).to(device)
    loss = criterion(pred, gt)

    loss.backward()
    optimizer.step()

    print(f"Epoch: {i:03d}, Loss: {loss:.4f}")
