# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from typing import Optional, Tuple

class ProjectionHead(nn.Module):
    """非线性投影头 (带BN版本)"""
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 use_bn: bool = True):
        super().__init__()
        
        # 构建MLP层
        layers = []
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class BaseEncoder(nn.Module):
    """基础编码器 (适配CIFAR的ResNet变体)"""
    def __init__(self,
                 arch: str = "resnet18",
                 pretrained: bool = False,
                 cifar_stem: bool = True):
        super().__init__()
        
        # 加载预定义模型
        if arch == "resnet18":
            base_model = resnet18(pretrained=pretrained)
        elif arch == "resnet50":
            base_model = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # 修改第一层适配CIFAR
        if cifar_stem:
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()  # 取消原最大池化
        
        # 移除最后的分类层
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 获取输出维度
        with torch.no_grad():
            dummy = torch.randn(2, 3, 32, 32)
            self.output_dim = self.feature_extractor(dummy).view(2, -1).shape[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return features.squeeze()

class SimCLR(nn.Module):
    """SimCLR整体模型"""
    def __init__(self,
                 encoder: nn.Module,
                 projection_head: nn.Module,
                 temperature: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = projection_head
        self.temperature = temperature
        self.criterion = ContrastiveLoss(temperature=temperature)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 提取特征
        h = self.encoder(x)
        # 计算投影
        z = self.projector(h)
        return h, z

    def compute_loss(self, z: torch.Tensor) -> torch.Tensor:
        """计算NT-Xent损失"""
        return self.criterion(z)

class ContrastiveLoss(nn.Module):
    """对比损失的高效实现 (支持分布式训练)"""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        
        # 生成标签矩阵
        labels = torch.cat([torch.arange(batch_size//2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        
        # 归一化特征
        z_norm = F.normalize(z, dim=1)
        similarity = torch.mm(z_norm, z_norm.T) / self.temperature
        
        # 排除对角线
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels = labels[~mask].view(batch_size, -1)
        similarity = similarity[~mask].view(batch_size, -1)
        
        # 正样本索引
        positives = similarity[labels.bool()].view(batch_size, -1)
        # 负样本索引
        negatives = similarity[~labels.bool()].view(batch_size, -1)
        
        # 构造logits
        logits = torch.cat([positives, negatives], dim=1)
        
        # 构造伪标签：正样本位于第0位置
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 计算交叉熵损失
        loss = self.criterion(logits, targets) / batch_size
        return loss

def build_model(config: dict) -> Tuple[nn.Module, nn.Module]:
    """模型构建工厂函数"""
    # 配置解析
    encoder_arch = config.get("encoder_arch", "resnet18")
    proj_hidden = config.get("proj_hidden", 256)
    proj_out = config.get("proj_out", 128)
    temperature = config.get("temperature", 0.5)
    use_bn = config.get("use_bn", True)
    
    # 构建编码器
    encoder = BaseEncoder(arch=encoder_arch)
    
    # 构建投影头
    projector = ProjectionHead(
        input_dim=encoder.output_dim,
        hidden_dim=proj_hidden,
        output_dim=proj_out,
        use_bn=use_bn
    )
    
    # 整合模型
    model = SimCLR(
        encoder=encoder,
        projection_head=projector,
        temperature=temperature
    )
    
    return model

# 测试代码
if __name__ == "__main__":
    # 测试配置
    config = {
        "encoder_arch": "resnet18",
        "proj_hidden": 256,
        "proj_out": 128,
        "temperature": 0.5,
        "use_bn": True
    }
    
    # 构建模型
    model = build_model(config)
    
    # 验证前向传播
    dummy_input = torch.randn(256, 3, 32, 32)  # 假设batch_size=128
    h, z = model(dummy_input)
    print(f"特征维度: {h.shape}, 投影维度: {z.shape}")  # 应输出 torch.Size([256, 512]) 和 torch.Size([256, 128])
    
    # 验证损失计算
    loss = model.compute_loss(z)
    print(f"损失值: {loss.item():.4f}")  # 应为合理浮点数