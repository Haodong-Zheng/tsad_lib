"""
早停机制 - 监控验证损失，不再改善时停止训练
"""
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path='checkpoints/best_model.pth'):
        """
        参数:
        patience: 等待验证损失不再改善的epoch数
        delta: 被视为改善的最小变化量
        path: 最佳模型保存路径
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        # 转换为负值以便最大化（损失越小越好）
        score = -val_loss
        
        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 检查是否有改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # 有改善时重置计数器
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        # print(f'验证损失下降 ({self.val_loss_min:.6f} → {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        