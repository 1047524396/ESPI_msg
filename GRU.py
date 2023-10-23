import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

torch.manual_seed(0)

data = np.load("vector_list.npy")
label = pd.read_csv("qume_part.csv")
X_train = torch.tensor(data, dtype=torch.float32)
y_train = torch.tensor(label['vulnerability'], dtype=torch.float32) 

# 定义一个全连接神经网络模型，2层
# class SimpleModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         output = self.fc2(x)
#         return self.sigmoid(output)
    
class SimpleModel(nn.Module):#4层
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, hidden_size)  
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x))  
        output = self.fc4(x)
        return self.sigmoid(output)


# 初始化模型
input_size = 128
hidden_size = 128
output_size = 1
model = SimpleModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
early_stopping_patience = 10  # 定义早停的耐心值
best_loss = float('inf')
no_improvement_count = 0

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train)
    loss.backward()
    optimizer.step()

    # 早停策略
    if loss < best_loss:
        best_loss = loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}.')
        break

input_data = X_train  # 使用训练数据进行预测

# 使用模型进行预测
with torch.no_grad():
    output = model(input_data)
    predicted_classes = torch.round(output).squeeze().numpy()

print("预测结果：", output , predicted_classes)
