import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader,Data
import pandas as pd
import numpy as np

# 修改嵌入层和GRU的维度
class GGNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GGNN, self).__init__()
        self.embedding_size = 128  # 嵌入层大小
        self.hidden_size = 128     # GRU隐藏状态大小
        self.conv1 = GraphConvolution(num_features, self.embedding_size)  # 创建第一个图卷积层
        self.conv2 = GraphConvolution(self.embedding_size, num_classes)   # 创建第二个图卷积层
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)  # GRU层

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return self.lin(x_j)

# 加载Cora数据集
# dataset = Planetoid(root='cora', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]

msg_data = pd.read_csv("qume_part.csv")
y = msg_data['vulnerability']
X = np.load("vector_list.npy")
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)
data = Data(x=x, y=y)

# 创建数据加载器
loader = DataLoader([data], batch_size=64, shuffle=True)

# 初始化模型和优化器
model = GGNN(num_features=x.size(1), num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率0.001

# 添加早停策略
best_accuracy = 0
early_stop_counter = 0
model.train()
for epoch in range(20):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

    # 评估准确度
    model.eval()
    correct = 0
    for batch in loader:
        out = model(batch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()
    accuracy = correct / len(X)
    print(f'Accuracy: {accuracy}')

    # 检查早停条件
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 10:
            print("Early stopping: No improvement for 10 epochs.")
            break

# 进行预测
# new_data = dataset[0]  # 替换为你自己的数据
# model.eval()
# pred = model(new_data).max(dim=1)[1]
# predicted_class = pred.argmax().item()
# print(f'Predicted class: {predicted_class}')
