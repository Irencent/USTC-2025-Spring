# dataloader.py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple, Optional, List, Any
import matplotlib.pyplot as plt
from PIL import Image

# 配置参数类（可单独保存为config.py）
class DataConfig:
    """数据加载配置参数"""
    def __init__(self, image_size: int = 32, 
                 seed: int = 42,
                 subset_classes: int = 10,
                 train_percent: float = 0.1, 
                 color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1),
                 gray_scale_prob: float = 0.2):
        
        self.image_size = image_size
        self.subset_classes = subset_classes
        self.train_percent = train_percent
        self.color_jitter = color_jitter
        self.gray_scale_prob = gray_scale_prob
        self.seed = seed

def numpy_augmentation(img: Image.Image) -> np.ndarray:
    return np.array(img)

def get_simclr_augmentation(config: DataConfig, normalize: bool = True) -> transforms.Compose:
    """生成SimCLR数据增强序列"""
    color_jitter = transforms.ColorJitter(
        brightness=config.color_jitter[0],
        contrast=config.color_jitter[1],
        saturation=config.color_jitter[2],
        hue=config.color_jitter[3],
    )
    
    transform_list = [
        transforms.RandomResizedCrop(
            size=config.image_size,
            scale=(0.08, 1.0),  # 原论文推荐的尺度范围
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=config.gray_scale_prob),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ))
    
    return transforms.Compose(transform_list)

class SimCLRDataset(Dataset):
    """生成SimCLR训练所需的数据对（两个增强视图）"""
    def __init__(self, dataset: Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取原始图像（假设dataset返回的是PIL.Image和标签）
        image, target = self.dataset[index]
        # 应用两次不同的增强
        view1 = self.transform(image)
        view2 = self.transform(image)
        return (view1, view2), target

def get_datasets(config: DataConfig) -> Tuple[Dataset, Dataset]:
    """获取训练集和测试集（已应用数据预处理）"""
    # 加载原始CIFAR-10数据集
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    test_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None
    )

    # 筛选指定类别
    if config.subset_classes < 10:
        # 筛选训练集
        train_targets = np.array(train_data.targets)
        train_indices = np.where(train_targets < config.subset_classes)[0]
        train_data = Subset(train_data, train_indices)
        # 筛选测试集
        test_targets = np.array(test_data.targets)
        test_indices = np.where(test_targets < config.subset_classes)[0]
        test_data = Subset(test_data, test_indices)

    # 从训练集中采样子集
    num_train = len(train_data)
    subset_size = int(config.train_percent * num_train)
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, _ = random_split(
        train_data, [subset_size, num_train - subset_size], generator=generator
    )

    # 应用SimCLR数据增强
    simclr_transform = get_simclr_augmentation(config)
    simclr_train_dataset = SimCLRDataset(train_subset, simclr_transform)

    # 测试集使用基础转换
    test_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    test_dataset = SimCLRDataset(test_data, test_transform)

    return simclr_train_dataset, test_dataset

def get_dataloaders(
    config: DataConfig, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """获取训练和测试的DataLoader"""
    train_dataset, test_dataset = get_datasets(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # SimCLR需要完整batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def plot_augmented_images(
    dataset: Dataset, 
    num_samples: int = 5, 
    figsize: Tuple[int, int] = (10, 4)
) -> None:
    """可视化数据增强后的图像对"""
    plt.figure(figsize=figsize)
    
    for i in range(num_samples):
        # 随机选择一个样本
        idx = np.random.randint(len(dataset))
        (view1, view2), _ = dataset[idx]
        
        # 反归一化显示图像
        view1 = inverse_normalize(view1).permute(1, 2, 0).numpy()
        view2 = inverse_normalize(view2).permute(1, 2, 0).numpy()
        
        # 绘制第一个增强视图
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(view1)
        plt.axis('off')
        if i == 0:
            plt.title('View 1')
        
        # 绘制第二个增强视图
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(view2)
        plt.axis('off')
        if i == 0:
            plt.title('View 2')
    
    plt.tight_layout()
    plt.show()

def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化操作"""
    inv_normalize = transforms.Normalize(
        mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
        std=[1/0.2023, 1/0.1994, 1/0.2010]
    )
    return inv_normalize(tensor)

# 示例用法
if __name__ == "__main__":
    config = DataConfig()
    train_loader, test_loader = get_dataloaders(config, batch_size=256)
    
    # 可视化增强效果
    #train_dataset, _ = get_datasets(config)
    #plot_augmented_images(train_dataset)
    images, labels = next(iter(train_loader))
    print(f"Batch size: {images[0].shape}, Labels: {labels.shape}")