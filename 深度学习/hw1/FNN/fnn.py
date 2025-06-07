from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# 加载数据集
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame
X = df[boston.feature_names].values
y = df['MEDV'].values.reshape(-1, 1)

# 划分训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 构建前馈神经网络

import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[64], activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # 添加隐藏层
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            prev_size = size
        
        # 输出层
        self.output = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x
    
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    best_loss = float('inf')
    best_weights = None
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    model.load_state_dict(best_weights)
    return train_losses, val_losses

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            test_loss += criterion(outputs, y_test).item()
    return test_loss / len(test_loader)

# 配置参数
configs = {
    "shallow": [64],
    "medium": [64, 64],
    "deep": [64, 64, 64]
}

results = {}
for depth, hidden_sizes in configs.items():
    print(f"\nTraining {depth} network")
    model = FeedForwardNN(hidden_sizes=hidden_sizes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    
    train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200)
    test_loss = evaluate_model(model, test_loader, criterion)
    
    results[depth] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }

learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_results = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = FeedForwardNN(hidden_sizes=[64, 64])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    test_loss = evaluate_model(model, test_loader, criterion)
    
    lr_results[lr] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }

activations = ['relu', 'tanh', 'sigmoid']
activation_results = {}

for act in activations:
    print(f"\nTraining with activation: {act}")
    model = FeedForwardNN(hidden_sizes=[64, 64], activation=act)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    test_loss = evaluate_model(model, test_loader, criterion)
    
    activation_results[act] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }

# 可视化结果
import matplotlib.pyplot as plt

def plot_results(history, title):
    plt.figure(figsize=(10, 6))
    for label, data in history.items():
        plt.plot(data['train_loss'], label=f'{label} Train')
        plt.plot(data['val_loss'], linestyle='--', label=f'{label} Val')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

# 网络深度对比
plot_results(results, "Network Depth Comparison")

# 学习率对比
plot_results(lr_results, "Learning Rate Comparison")

# 激活函数对比
plot_results(activation_results, "Activation Function Comparison")