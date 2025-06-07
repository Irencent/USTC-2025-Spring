import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # d_model 是嵌入维度
        # 计算位置编码
        # 位置编码公式: PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        # 这里的pos是位置，i是维度索引
        position = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ) # (d_model/2,)
        # 计算sin和cos
        # 位置编码公式: PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 位置编码公式: PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加batch维度，pe的形状是(max_len, d_model)，我们需要添加一个batch维度，使其形状为(1, max_len, d_model)，这样在forward方法中可以直接与输入x相加
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, : x.size(1)]

class DIYMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_head整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义投影矩阵
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: (batch_size, seq_len, embed_dim)
        输出形状: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                 mask: torch.Tensor = None) -> torch.Tensor:
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 生成下三角掩码（包含对角线）
        seq_len = Q.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device), 
            diagonal=1
        ).bool()  # 形状: (seq_len, seq_len)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 应用mask（用于padding）
        if mask is not None:
            # mask形状: (batch_size, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算上下文向量
        context = torch.matmul(attn_weights, V)
        return context, attn_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # 投影到Q/K/V空间
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割多头
        Q = self._split_heads(Q)  # (batch_size, num_heads, seq_len, head_dim)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 计算注意力
        context, attn_weights = self._attention(Q, K, V, key_padding_mask)
        
        # 合并多头
        batch_size, _, seq_len, _ = context.size() # (batch_size, num_heads, seq_len, head_dim)
        # 先转置到(batch_size, seq_len, num_heads, head_dim)，再reshape到(batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 最终投影
        output = self.W_o(context) # output形状: (batch_size, seq_len, embed_dim)
        return output, attn_weights # attn_weights形状: (batch_size, num_heads, seq_len, seq_len)
        
        
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_classes: int = 1, # 使用交叉熵损失为2，二元交叉熵为1
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        # 自定义多头注意力
        self.attention = DIYMultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1
        )
        # 分类器
        # nn.Sequential是一个容器模块，可以将多个层组合在一起
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, num_classes),
        )
        self.attentions = []  # 新增存储注意力权重的属性

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # 添加位置编码
        x = self.pos_encoder(x) * math.sqrt(self.embedding.embedding_dim)
        
        # 生成自注意力掩码（忽略填充位置）,这里我们使用sum(dim=-1)来判断填充位置，sum(dim=-1)的结果是(batch_size, seq_len)，如果某个位置是填充位置，则该位置的和为0
        padding_mask = (x.sum(dim=-1) == 0)
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )

        self.attentions = attn_weights.detach().cpu()  # 存储为 (batch, heads, seq, seq)
        
        # 全局平均池化
        pooled = attn_output.mean(dim=1)  # (batch_size, d_model)
        return self.classifier(pooled)
    
    def get_attention_weights(self):
        return self.attentions

class SimpleRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 手动实现RNN参数
        # 输入到隐藏的权重和偏置
        self.Wxh = nn.ParameterList([
            nn.Parameter(torch.Tensor(embed_dim if l == 0 else hidden_dim, hidden_dim))
            for l in range(num_layers)
        ])
        # 隐藏到隐藏的权重
        self.Whh = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            for l in range(num_layers)
        ])
        self.bh = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim))
            for l in range(num_layers)
        ])
        
        # 初始化参数
        for layer in range(num_layers):
            # 使用Kaiming初始化输入到隐藏的权重和偏置，使用正交初始化隐藏到隐藏的权重
            # Kaiming初始化适用于ReLU激活函数，正交初始化适用于tanh激活函数
            nn.init.kaiming_normal_(self.Wxh[layer], mode='fan_in', nonlinearity='tanh') 
            nn.init.orthogonal_(self.Whh[layer])
            nn.init.zeros_(self.bh[layer])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 嵌入层
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 初始化隐藏状态
        hiddens = [torch.zeros(batch_size, self.hidden_dim, device=x.device) 
                  for _ in range(self.num_layers)]
        
        # RNN计算过程
        for t in range(seq_len):
            new_hiddens = []
            input_t = x[:, t, :]  # 当前时间步输入
            
            for layer in range(self.num_layers):
                # 计算隐藏状态
                h = hiddens[layer]
                Wxh = self.Wxh[layer]
                Whh = self.Whh[layer]
                bh = self.bh[layer]
                
                # 线性变换 + 非线性激活
                # 计算公式: h_next = tanh(Wxh[layer] * input_t + Whh[layer] * hiddens[layer] + bh[layer])
                h_next = torch.tanh(
                    torch.mm(input_t, Wxh) + 
                    torch.mm(h, Whh) + 
                    bh
                )
                
                # 更新输入用于下一层
                input_t = h_next
                new_hiddens.append(h_next)
            
            hiddens = new_hiddens
        
        # 取最后一层的最后时间步隐藏状态
        last_hidden = hiddens[-1]  # (batch_size, hidden_dim)
        
        return self.classifier(last_hidden)

    def init_hidden(self, batch_size):
        """初始化隐藏状态（已集成到forward中）"""
        return torch.zeros(batch_size, self.hidden_dim)

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 输入门参数
        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # 遗忘门参数
        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # 细胞更新参数
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        
        # 输出门参数
        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self._init_weights()

    def _init_weights(self):
        # Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                # 对于权重矩阵，使用Xavier初始化
                # Xavier初始化适用于tanh激活函数
                # 计算公式: W ~ U[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
                # 这里的fan_in和fan_out分别是输入和输出的维度
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x, init_states=None):
        """输入形状: (batch_size, seq_len, input_size)"""
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态和细胞状态
        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size).to(x.device))
        if init_states is not None:
            h_t, c_t = init_states
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入
            
            # --- 核心计算部分 ---
            # 输入门
            # 计算公式: i_t = σ(W_xi * x_t + W_hi * h_t + b_i)
            # 这里的x_t是当前时间步的输入，h_t是上一时间步的隐藏状态
            i_t = torch.sigmoid(
                torch.mm(x_t, self.W_xi) + 
                torch.mm(h_t, self.W_hi) + 
                self.b_i
            )
            
            # 遗忘门
            # 计算公式: f_t = σ(W_xf * x_t + W_hf * h_t + b_f)
            # 遗忘门决定了细胞状态中哪些信息需要被遗忘
            f_t = torch.sigmoid(
                torch.mm(x_t, self.W_xf) + 
                torch.mm(h_t, self.W_hf) + 
                self.b_f
            )
            
            # 细胞状态更新
            c_hat_t = torch.tanh(
                torch.mm(x_t, self.W_xc) + 
                torch.mm(h_t, self.W_hc) + 
                self.b_c
            )
            # 细胞状态更新公式: c_t = f_t * c_t + i_t * c_hat_t
            # 这里的c_t是上一时间步的细胞状态，i_t是当前时间步的输入门
            # c_hat_t是当前时间步的候选细胞状态
            c_t = f_t * c_t + i_t * c_hat_t
            
            # 输出门
            o_t = torch.sigmoid(
                torch.mm(x_t, self.W_xo) + 
                torch.mm(h_t, self.W_ho) + 
                self.b_o
            )
            
            # 隐藏状态更新
            h_t = o_t * torch.tanh(c_t)
            
            outputs.append(h_t.unsqueeze(1))
        
        return torch.cat(outputs, dim=1), (h_t, c_t)
    
class DIYLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = SimpleLSTM(input_size=embed_dim, hidden_size=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM前向传播
        outputs, (h_n, c_n) = self.lstm(x)  # outputs形状: (batch_size, seq_len, hidden_dim)
        
        # 取最后时间步的隐藏状态
        last_hidden = h_n  # 或 outputs[:, -1, :]
        return self.classifier(last_hidden)