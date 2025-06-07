import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

def test_evaluation(model, test_loader):
    _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 混淆矩阵
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_curves(results, title):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for label, data in results.items():
        plt.plot(data['train_loss'], label=f'{label} Train')
        plt.plot(data['val_loss'], linestyle='--', label=f'{label} Val')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, data in results.items():
        plt.plot(data['train_acc'], label=f'{label} Train')
        plt.plot(data['val_acc'], linestyle='--', label=f'{label} Val')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载完整数据集
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform
)

# 随机抽取1%数据（可修改ratio调整采样比例）
def get_subset(dataset, ratio=0.01):
    indices = np.random.choice(
        len(dataset), 
        size=int(len(dataset)*ratio), 
        replace=False
    )
    return Subset(dataset, indices)

# 获取子集
train_subset = get_subset(train_dataset, ratio=0.01)  # 600 samples
test_subset = get_subset(test_dataset, ratio=0.01)    # 100 samples

# 划分训练集和验证集（8:2比例）
val_ratio = 0.2
train_size = int(len(train_subset) * (1 - val_ratio))
val_size = len(train_subset) - train_size

train_subset, val_subset = random_split(
    train_subset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 创建DataLoader
batch_size = 32
train_loader = DataLoader(
    train_subset, 
    batch_size=batch_size, 
    shuffle=True,  # 训练集需要shuffle
    num_workers=2
)
val_loader = DataLoader(
    val_subset, 
    batch_size=batch_size, 
    shuffle=False, # 验证集不需要shuffle
    num_workers=2
)
test_loader = DataLoader(
    test_subset, 
    batch_size=batch_size, 
    shuffle=False, # 测试集不需要shuffle
    num_workers=2
)

class FlexibleCNN(nn.Module):
    def __init__(self, depths=[1,1], kernel_sizes=[3,3], num_classes=10):
        super().__init__()
        layers = []
        in_channels = 1
        
        # 修改池化层为自适应池化
        for depth, kernel_size in zip(depths, kernel_sizes):
            for _ in range(depth):
                layers += [
                    nn.Conv2d(in_channels, 32, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(14)  # 统一输出为14x14
                ]
                in_channels = 32
        
        # 最后添加全局平均池化
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(32, num_classes)  # 输入维度固定为32

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train_model(model, train_loader, val_loader, config_params, num_epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100*correct/total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'config': config_params  # 保存结构参数
            }, 'best_model.pth')

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%\n")
    
    return history

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss/len(loader), 100*correct/total




if __name__ == '__main__':

    # 网络深度对比实验配置
    depth_configs = {
        "shallow": [1, 0],      # 1个卷积层
        "medium": [2, 1],       # 3个卷积层
        "deep": [3, 2]          # 5个卷积层
    }

    # 卷积核对比实验配置
    kernel_configs = {
        "small": [3, 3],        # 3x3卷积核
        "medium": [5, 5],       # 5x5卷积核
        "large": [7, 7]         # 7x7卷积核
    }

    # 执行深度实验
    depth_results = {}
    for name, config in depth_configs.items():
        print(f"\nTraining {name} network")
        model_config = {
            'depths': config,
            'kernel_sizes': [3, 3],  # 深度实验固定使用3x3卷积核
            'num_classes': 10
        }
        model = FlexibleCNN(**model_config)
        # 将配置参数传给训练函数
        history = train_model(model, train_loader, val_loader, config_params=model_config)
        depth_results[name] = history

    # 执行卷积核实验
    kernel_results = {}
    for name, config in kernel_configs.items():
        print(f"\nTraining with {name} kernels")
        model_config = {
            'depths': [2, 1],        # 卷积核实验固定使用[2,1]深度
            'kernel_sizes': config,
            'num_classes': 10
        }
        model = FlexibleCNN(**model_config)
        # 将配置参数传给训练函数
        history = train_model(model, train_loader, val_loader, config_params=model_config)
        kernel_results[name] = history

    # 可视化深度实验结果
    plot_curves(depth_results, "Network Depth Comparison")

    # 可视化卷积核实验结果
    plot_curves(kernel_results, "Kernel Size Comparison")

    # # 执行测试评估
    best_model_info = torch.load('best_model.pth', weights_only=True)  # 添加安全加载
    best_model = FlexibleCNN(**best_model_info['config'])
    best_model.load_state_dict(best_model_info['state_dict'])
    test_evaluation(best_model, test_loader)