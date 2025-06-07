import torch
import torch.nn as nn
from model  import MultiHeadAttentionClassifier, SimpleRNNClassifier, DIYLSTMClassifier
from train import Config
from prettytable import PrettyTable
from load_dataset_hys import load_and_preprocess_data, build_vocab, vectorize_text

import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, dataloader, config, sample_idx=0):
    """可视化指定样本的注意力权重"""
    # 1. 获取样本数据
    model.eval()
    batch = next(iter(dataloader))  # 取第一个batch
    inputs, labels = batch
    sample_input = inputs[sample_idx:sample_idx+1].to(config.device)  # (1, seq_len)
    
    # 2. 获取注意力权重
    with torch.no_grad():
        outputs = model(sample_input)
        # 需要修改模型使其返回注意力权重
        # 假设attn_weights的形状为 (num_layers, num_heads, seq_len, seq_len)
        all_attentions = model.get_attention_weights()  
    
    # 3. 获取文本tokens
    idx_to_word = {v:k for k,v in vocab.items()}  # vocab来自预处理
    tokens = [idx_to_word.get(idx.item(), "<unk>") for idx in sample_input[0]]
    
    # 4. 可视化单个层的多头注意力
    layer = 0  # 观察第一个注意力层
    attn_weights = all_attentions[layer][0]  # (num_heads, seq_len, seq_len)
    
    # 创建多子图
    num_heads = attn_weights.size(0)
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    
    for head in range(num_heads):
        # 绘制热力图
        ax = axes[head]
        im = ax.imshow(attn_weights[head].cpu().numpy(), cmap='viridis')
        ax.set_title(f"Head {head+1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def count_parameters(model, verbose=False):
    """计算模型参数量
    Args:
        model: PyTorch模型
        verbose: 是否打印详细分层统计
    Returns:
        total_params: 总参数量（单位：百万）
    """
    table = PrettyTable(["Layer", "Parameters", "Shape"])
    total_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        params = param.numel()
        total_params += params
        if verbose:
            table.add_row([name, f"{params:,}", str(param.shape)])
    
    if verbose:
        print(table)
        print(f"\nTotal Trainable Params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return total_params

if __name__ == "__main__":
    # 初始化配置
    config = Config()

    df = load_and_preprocess_data(config.data_path)
    vocab = build_vocab(df['text'], config.max_vocab_size)
    # 初始化不同模型
    models = {
        "MHA": MultiHeadAttentionClassifier(
            vocab_size=len(vocab),
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len
        ),
        "RNN": SimpleRNNClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_layers=2,
            num_classes=2,
        ),
        "LSTM": DIYLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=config.d_model,
            hidden_dim=config.d_model,
            num_classes=2,
        )  # 使用之前定义的Transformer
    }

    # 打印各模型参数量对比
    comp_table = PrettyTable(["Model", "Parameters", "Size (MB)"])
    for name, model in models.items():
        params = count_parameters(model)
        comp_table.add_row([
            name,
            f"{params/1e6:.2f}M",
            f"{(params * 4)/1e6:.2f}MB"  # 假设float32存储(4字节/参数)
        ])
    
    print("\n模型参数量对比:")
    print(comp_table)

    # 打印MultiHeadAttentionClassifier详细参数分布
    print("\nMultiHeadAttentionClassifier参数详情:")
    count_parameters(models["MHA"], verbose=True)

    # 打印SimpleRNNClassifier详细参数分布
    print("\nSimpleRNNClassifier参数详情:")
    count_parameters(models["RNN"], verbose=True)
    # 打印DIYLSTMClassifier详细参数分布 
    print("\nDIYLSTMClassifier参数详情:")
    count_parameters(models["LSTM"], verbose=True)

    

    # 使用示例
    visualize_attention(model, test_loader, config, sample_idx=5)