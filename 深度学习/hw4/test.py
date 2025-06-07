# test.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from model import build_model
from dataloader import get_datasets, DataConfig

def load_config_and_model(config_path, checkpoint_path):
    """加载配置和模型（自动适配编码器输出维度）"""
    config = torch.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建原始模型结构
    model = build_model(config["model"]).to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    
    # 获取编码器及其输出维度
    encoder = model.encoder
    encoder.output_dim = encoder.output_dim  # 继承BaseEncoder中计算的维度
    
    return config, encoder, device

def train_classifier(encoder, train_loader, device, num_classes, epochs=10):
    """训练线性分类头（自动获取编码器输出维度）"""
    encoder.eval()
    
    # 动态获取编码器输出维度
    classifier_input_dim = encoder.output_dim  
    
    classifier = torch.nn.Linear(classifier_input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images[0].to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Classifier Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return classifier

def evaluate_model(encoder, classifier, test_loader, device):
    """评估模型性能"""
    encoder.eval()
    classifier.eval()
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images[0].to(device)
            
            # 提取特征
            features = encoder(images)
            
            # 分类预测
            outputs = classifier(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并结果
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_preds, all_labels

def main():
    # 路径配置（根据实际训练保存路径修改）
    config_path = "./saved_models/exp_20250529_113345/config.pth"
    checkpoint_path = "./saved_models/exp_20250529_113345/best_model.pth"
    
    # 加载模型
    config, encoder, device = load_config_and_model(config_path, checkpoint_path)
    
    # 数据配置（与训练时一致）
    data_config = DataConfig(
        image_size=config["data"]["image_size"],
        subset_classes=config["data"]["subset_classes"],
        train_percent=0.8  # 80%数据用于训练分类头
    )
    
    # 获取数据集
    train_dataset, test_dataset = get_datasets(data_config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )
    
    # 训练分类头
    num_classes = config["data"]["subset_classes"]
    classifier = train_classifier(encoder, train_loader, device, num_classes)
    
    # 评估
    probs, preds, labels = evaluate_model(encoder, classifier, test_loader, device)
    
    # 计算指标
    accuracy = accuracy_score(labels, preds)
    y_true_bin = np.eye(num_classes)[labels.astype(int)]
    auc = roc_auc_score(y_true_bin, probs, multi_class='ovr', average='macro')
    
    print(f"\nEvaluation Results:")
    print(f"- Test Accuracy: {accuracy:.4f}")
    print(f"- Macro AUC: {auc:.4f}")

if __name__ == "__main__":
    main()