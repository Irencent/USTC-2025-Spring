import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from load_dataset_hys import load_and_preprocess_data, build_vocab, vectorize_text, create_dataloaders
from model import MultiHeadAttentionClassifier, SimpleRNNClassifier, DIYLSTMClassifier
from sklearn.metrics import f1_score, recall_score

import matplotlib.pyplot as plt
import numpy as np

# 训练配置
class Config:
    # 数据参数
    data_path = 'enron_spam_data.csv'
    max_seq_len = 200
    batch_size = 64
    
    # 模型参数
    model_name = 'MultiHeadAttentionClassifier'  # 可选 'MultiHeadAttentionClassifier', 'SimpleRNNClassifier', 'DIYLSTMClassifier'
    d_model = 128
    n_heads = 4
    max_vocab_size = 20000
    
    # 训练参数
    lr = 3e-4
    epochs = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = f'{model_name}_best_model_{max_seq_len}.pth'

# 训练函数
def train(model, train_loader, test_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用二进制交叉熵损失函数
    #criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            # 做法是将所有参数的梯度裁剪到一个范围内，这里使用1.0作为裁剪的最大范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            # 使用交叉熵函数时
            _, predicted = outputs.max(1)
            # 使用二进制交叉熵函数时    
            #predicted = (torch.sigmoid(outputs) > 0.5).float()
            # 计算正确预测的数量
            # predicted是模型的预测结果，labels是实际标签
            # predicted.eq(labels)返回一个布尔张量，表示预测是否正确
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # 验证阶段
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, config)
        
        # 打印统计信息
        train_loss = epoch_loss / total
        train_acc = correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch: {epoch+1:02} | Time: {epoch_time:.2f}s")
        print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"New best model saved! (Acc: {best_acc*100:.2f}%)")

# 评估函数
def evaluate(model, loader, criterion, config):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算预测结果
            _, predicted = torch.max(outputs, 1)
            # 使用二进制交叉熵函数时
            #predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            # 累计统计量
            epoch_loss += loss.item() * inputs.size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 收集所有预测和标签（移动到CPU）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各项指标
    avg_loss = epoch_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary')  # 二分类使用binary
    recall = recall_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, f1, recall

def visualize_attention(model, dataloader, config, sample_idx=0):
    """可视化指定样本的注意力权重"""
    model.eval()
    batch = next(iter(dataloader))
    inputs, labels = batch
    sample_input = inputs[sample_idx:sample_idx+1].to(config.device)  # 确保取到正确的样本
    
    with torch.no_grad():
        outputs = model(sample_input)
        all_attentions = model.get_attention_weights()  # 形状应为 (1, num_heads, seq_len, seq_len)
    
    # 确保处理单个样本的注意力权重
    attn_weights = all_attentions[0]  # 形状 (num_heads, seq_len, seq_len)
    
    idx_to_word = {v: k for k, v in vocab.items()}
    tokens = [idx_to_word.get(idx.item(), "<unk>") for idx in sample_input[0]]
    
    num_heads = attn_weights.size(0)
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    
    for head in range(num_heads):
        ax = axes[head] if num_heads > 1 else axes
        # 确保每个头的权重是二维的
        head_weights = attn_weights[head].cpu().numpy()
        im = ax.imshow(head_weights, cmap='viridis')
        ax.set_title(f"Head {head+1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 初始化配置
    config = Config()
    
    # 加载和预处理数据
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(config.data_path)
    vocab = build_vocab(df['text'], config.max_vocab_size)
    X_sequences, y = vectorize_text(df, vocab, config.max_seq_len)
    train_loader, val_loader, test_loader = create_dataloaders(X_sequences, y, config.batch_size)
    
    # 初始化模型
    if config.model_name == 'MultiHeadAttentionClassifier':
        model = MultiHeadAttentionClassifier(
            vocab_size=len(vocab),
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == 'SimpleRNNClassifier':
        model = SimpleRNNClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_layers=2,
            num_classes=2,
        )
    else:
        model = DIYLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_classes=2,
        )
    # 打印模型参数
    print(f"Model architecture:\n{model}")
    print(f"Using device: {config.device}")
    
    # 开始训练
    print("Starting training...")
    train(model, train_loader, val_loader, config)

    # 训练完成后，加载最佳模型进行评估
    if config.model_name == 'MultiHeadAttentionClassifier':
        best_model = MultiHeadAttentionClassifier(
            vocab_size=len(vocab),
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == 'SimpleRNNClassifier':
        best_model = SimpleRNNClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_layers=2,
            num_classes=2,
        )
    else:
        best_model = DIYLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_classes=2,
        )
    # 加载最佳模型参数
    best_model.load_state_dict(torch.load(config.save_path, weights_only=True))
    best_model = best_model.to(config.device)
    test_loss, test_acc, test_f1, test_recall = evaluate(best_model, test_loader, nn.CrossEntropyLoss(), config)
    print(f"Best model test loss: {test_loss:.4f} | test accuracy: {test_acc*100:.2f}%| test F1: {test_f1:.4f} | test Recall: {test_recall:.4f}")
