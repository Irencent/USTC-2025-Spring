import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# 下载NLTK资源
nltk.download('stopwords')

# 1. 数据加载与预处理
def load_and_preprocess_data(filepath):
    # 加载数据
    # 明确指定关键列为字符串类型，其他列不加载
    df = pd.read_csv(
        filepath,
        usecols=['Date', 'Subject', 'Message', 'Spam/Ham'],  # 只加载必要列
        dtype={'Subject': str, 'Message': str},  # 强制文本列类型
        parse_dates=['Date'],  # 明确日期解析
        low_memory=False
    ).rename(columns={'Spam/Ham': 'label'})
    
    # 删除缺失值
    df.dropna(inplace=True)
    
    # 过滤无效标签
    df = df[df['label'].isin(['ham', 'spam'])]
    
    # 合并文本字段
    df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
    df = df[['text', 'label']]
    
    # 转换标签
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 添加空文本检查
    empty_text_mask = df['text'].str.strip() == ''
    print(f"空文本数量: {empty_text_mask.sum()}")
    df = df[~empty_text_mask]  # 过滤空文本
    
    return df

# 2. 文本预处理增强
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess(self, text):
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        words = text.split()
        
        # 移除停用词并将单词还原为词干形式
        processed_words = [
            self.stemmer.stem(word) 
            for word in words 
            if word not in self.stop_words
        ]
        
        return ' '.join(processed_words)

def build_vocab(text_series, max_vocab_size=10000):
    """构建词表"""
    token_counter = Counter()  # 初始化计数器
    
    # 统计所有文本中的词频
    for text in text_series:
        tokens = text.split()
        token_counter.update(tokens)  # 更新词频
    
    # 保留高频词（Top N）
    vocab_items = token_counter.most_common(max_vocab_size)
    
    # 构建词表字典
    vocab = {
        "<pad>": 0,  # 填充符
        "<unk>": 1,  # 未知词
        **{word: idx + 2 for idx, (word, _) in enumerate(vocab_items)}  # 从2开始索引
    }
    return vocab

# 3. 文本向量化
def vectorize_text(df, vocab, max_seq_len=200):
    """将文本转换为词索引序列"""
    sequences = []
    for text in df['text']:
        tokens = text.split()

        truncated = tokens[-max_seq_len:]  # 取最后200个token
        # 处理超过最大长度的文本
        pad_num = max_seq_len - len(truncated)
        seq = [vocab["<pad>"]] * pad_num + [  # 左侧填充
            vocab.get(token, vocab["<unk>"]) 
            for token in truncated
        ]
        
        sequences.append(seq)
    return torch.LongTensor(sequences), \
        df['label'].values.astype(np.int64)  
# 4. 数据集划分与PyTorch数据加载
def create_dataloaders(X_sequences, y_labels, batch_size=32, val_ratio=0.2):
    """创建训练、验证和测试数据加载器"""
    X_tensor = torch.LongTensor(X_sequences)
    y_tensor = torch.LongTensor(y_labels)
    
    # 第一次划分：训练+验证 与 测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_tensor, y_tensor,
        test_size=0.2,
        stratify=y_labels,
        random_state=42
    )
    
    # 第二次划分：训练集 与 验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=42
    )
    
    # 创建三个数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # 创建三个DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# 主流程
if __name__ == "__main__":
    # 加载数据
    df = load_and_preprocess_data('enron_spam_data.csv')
    # 构建词表
    vocab = build_vocab(df['text'], max_vocab_size=20000)
    print(f"词表大小: {len(vocab)}")
    
    # 生成词索引序列
    X_sequences, y = vectorize_text(df, vocab, max_seq_len=200)
    
    # 创建数据加载器
    train_loader, val_loader,test_loader = create_dataloaders(X_sequences, y)
    
    # 验证数据形状
    for batch in train_loader:
        inputs, labels = batch
        print(f"输入形状: {inputs.shape}")  # 应为 (batch_size, seq_len)
        print(f"标签形状: {labels.shape}")  # 应为 (batch_size,)
        break