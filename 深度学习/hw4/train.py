# train.py
import os
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from model import build_model
from dataloader import get_datasets, get_dataloaders, DataConfig

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_components()
        self._prepare_logging()
        
    def _init_components(self):
        """初始化模型、优化器、数据加载器等"""
        # 构建模型
        self.model = build_model(self.config["model"]).to(self.device)
        
        # 优化器设置
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"]
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["train"]["epochs"],
            eta_min=self.config["optim"]["min_lr"]
        )
        
        # 混合精度训练
        self.scaler = amp.GradScaler(enabled=self.config["train"]["use_amp"])
        
        # 数据加载
        self.train_loader, _ = get_dataloaders(
            DataConfig(),
            batch_size=self.config["data"]["batch_size"],
        )
        
    def _prepare_logging(self):
        """准备日志和保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            self.config["train"]["save_dir"],
            f"exp_{timestamp}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # TensorBoard记录
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))
        
        # 保存配置文件
        torch.save(self.config, os.path.join(self.save_dir, "config.pth"))
        
    def _save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        
        filename = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(state, filename)
        
        if is_best:
            torch.save(state, os.path.join(self.save_dir, "best_model.pth"))
            
    def _train_epoch(self, epoch):
        """单个epoch的训练"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (views, _) in enumerate(progress_bar):
            # 移动数据到设备
            inputs = torch.cat(views, dim=0).to(self.device)  # [2B, C, H, W]
            
            # 混合精度训练上下文
            with amp.autocast(enabled=self.config["train"]["use_amp"]):
                # 前向传播
                _, projections = self.model(inputs)
                loss = self.model.compute_loss(projections)
                
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config["optim"]["grad_clip"] is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["optim"]["grad_clip"]
                )
                
            # 参数更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # TensorBoard记录
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/loss", loss.item(), global_step)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], global_step)
            
        return total_loss / len(self.train_loader)
    
    def train(self):
        """完整的训练流程"""
        best_loss = float("inf")
        
        for epoch in range(1, self.config["train"]["epochs"] + 1):
            # 训练一个epoch
            avg_loss = self._train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存检查点
            if epoch % self.config["train"]["save_interval"] == 0:
                self._save_checkpoint(epoch)
                
            # 更新最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(epoch, is_best=True)
                
            # 打印日志
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}")
            
        self.writer.close()

if __name__ == "__main__":
    # 训练配置示例
    config = {
        "data": 
        {
            "image_size": 32,
            "subset_classes": 10,
            "train_percent": 0.1,
            "batch_size": 256,
            "num_workers": 4,
            "color_jitter": (0.4, 0.4, 0.4, 0.1),
            "gray_scale_prob": 0.2
        },
        
        "model": {
            "encoder_arch": "resnet18",
            "proj_hidden": 256,
            "proj_out": 128,
            "temperature": 0.5,
            "use_bn": True
        },
        "optim": {
            "lr": 1e-3,
            "min_lr": 1e-5,
            "weight_decay": 1e-4,
            "grad_clip": 1.0
        },
        "train": {
            "epochs": 20,
            "use_amp": True,
            "save_dir": "./saved_models",
            "save_interval": 10
        }
    }
    
    # 初始化训练器并开始训练
    trainer = Trainer(config)
    trainer.train()