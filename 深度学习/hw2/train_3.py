import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from load_dataset_hys import load_and_preprocess_data, build_vocab, vectorize_text, create_dataloaders
from model import MultiHeadAttentionClassifier, SimpleRNNClassifier, DIYLSTMClassifier

# 训练配置
class Config:
    # 数据参数
    data_path = 'enron_spam_data.csv'
    max_seq_len = 200
    batch_size = 64
    
    # 模型参数
    model_name = 'DIYLSTMClassifier'  # 可选 'MultiHeadAttentionClassifier', 'SimpleRNNClassifier', 'DIYLSTMClassifier'
    d_model = 128
    n_heads = 4
    max_vocab_size = 20000
    
    # 训练参数
    lr = 3e-4
    epochs = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = f'{model_name}_best_model.pth'

# 训练函数
def train(model, train_loader, test_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
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
            _, predicted = outputs.max(1)
            # 计算正确预测的数量
            # predicted是模型的预测结果，labels是实际标签
            # predicted.eq(labels)返回一个布尔张量，表示预测是否正确
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # 验证阶段
        test_loss, test_acc = evaluate(model, test_loader, criterion, config)
        
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
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算accuracy , precision
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    
    return epoch_loss / total, correct / total

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
    best_model.load_state_dict(torch.load(config.save_path))
    best_model = best_model.to(config.device)
    test_loss, test_acc = evaluate(best_model, test_loader, nn.CrossEntropyLoss(), config)
    print(f"Best model test loss: {test_loss:.4f} | test accuracy: {test_acc*100:.2f}%")