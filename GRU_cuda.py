import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from load_dataset import load_fold_dataset, load_whole_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")

torch.manual_seed(0)

train_data, test_data = load_fold_dataset("5_fold_datasets", "spidb", 0)

train = np.load("train_vector.npy")
test = np.load("test_vector.npy")
#label = pd.read_csv("qume_part.csv")
X_train = torch.tensor(train, dtype=torch.float32)
y_train = torch.tensor(train_data['label'], dtype=torch.float32) 

X_test = torch.tensor(test, dtype=torch.float32)
y_test = torch.tensor(test_data['label'], dtype=torch.float32) 
    
# data = np.load("vector_list.npy")
# label = pd.read_csv("qume_part.csv")
# X_train = torch.tensor(data, dtype=torch.float32)
# y_train = torch.tensor(label['vulnerability'], dtype=torch.float32) 

# class SimpleModel(nn.Module): #4层
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)  
#         self.fc3 = nn.Linear(hidden_size, hidden_size)  
#         self.fc4 = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x)) 
#         x = torch.relu(self.fc3(x))  
#         output = self.fc4(x)
#         return self.sigmoid(output)
    
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return self.sigmoid(output)
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_hidden_size, gru_num_layers):
        super(GRUModel, self).__init__()
        
        # Add GRU layer
        self.gru = nn.GRU(input_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply the GRU layer
        gru_output, _ = self.gru(x)
        
        # Pass the output of the GRU through fully connected layers
        x = torch.relu(self.fc1(gru_output))
        output = self.fc2(x)
        return self.sigmoid(output)
# 初始化模型
input_size = 128
hidden_size = 128
output_size = 1
gru_hidden_size = 128 # Define GRU hidden size
gru_num_layers = 1  # Define the number of GRU layers
model = GRUModel(input_size, hidden_size, output_size, gru_hidden_size, gru_num_layers)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
early_stopping_patience = 10  # 定义早停的耐心值
best_loss = float('inf')
no_improvement_count = 0

for epoch in tqdm(range(num_epochs)):
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

input_data = X_test  # 使用训练数据进行预测

# 使用模型进行预测
with torch.no_grad():
    output = model(input_data)
    predicted_classes = torch.round(output).squeeze().numpy()

from sklearn.metrics import classification_report
acc = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in tqdm(range(len(predicted_classes)), desc="Processing"):
    if(y_test[i] == predicted_classes[i]):
        acc = acc + 1
    if(y_test[i] == 1 and predicted_classes[i] == 1):
        tp = tp + 1
    if(y_test[i]== 1 and predicted_classes[i] == 0):
        fp = fp + 1
    if(y_test[i] == 0 and predicted_classes[i] == 0):
        tn = tn + 1
    if(y_test[i] == 0 and predicted_classes[i] == 1):
        fn = fn + 1

# for i in tqdm(range(len(predicted_classes)), desc="Processing"):
#     if(y_train[i] == predicted_classes[i]):
#         acc = acc + 1
#     if(y_train[i] == 1 and predicted_classes[i] == 1):
#         tp = tp + 1
#     if(y_train[i]== 1 and predicted_classes[i] == 0):
#         fp = fp + 1
#     if(y_train[i] == 0 and predicted_classes[i] == 0):
#         tn = tn + 1
#     if(y_train[i] == 0 and predicted_classes[i] == 1):
#         fn = fn + 1
y_true = y_test  # 真实标签
y_pred = predicted_classes  # 模型的预测结果
report = classification_report(y_true, y_pred)
print(report)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = (2 * precision * recall) / ( precision + recall)
print("acc_rate:",acc/len(predicted_classes))
print("recall:",recall)
print("precision:",precision)
print("F1-score:",f1)
print("预测结果：", output , predicted_classes, len (output))
