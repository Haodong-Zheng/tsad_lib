import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from configs import Config

class WindowedDataset(Dataset):
    """创建滑动窗口数据集（重叠窗口）"""
    def __init__(self, data, window_size=Config.seq_len, stride=Config.stride):
        self.data = self.data = torch.from_numpy(data.astype(np.float32))
        self.window_size = window_size
        self.stride = stride
        self.length = (len(data) - window_size) // stride + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return window

class TestDataset(Dataset):
    def __init__(self, data, labels, window_size=100, stride=1):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.length = (len(data) - window_size) // stride + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        # return self.data[start:end], self.labels[end-1]  # 只返回最后一个时间点的标签
        return torch.from_numpy(self.data[start:end]), torch.tensor(self.labels[start])
    
def load_data(config):
    # 加载预处理数据
    if Config.dataset == "SDC":
        data_path = f"data/sdc/sdc_0802.npz"
    if Config.dataset == "MSL":
        data_path = f"data/MSL/msl.npz"
    if Config.dataset == "SMAP":
        data_path = f"data/SMAP/smap.npz"
    if Config.dataset == "SMD":
        data_path = f"data/SMD/smd.npz"
    if Config.dataset == "SWaT":
        data_path = f"data/SWaT/SWaT.npz"
    if Config.dataset == "PSM":
        data_path = f"data/PSM/psm.npz"
    if Config.dataset == "PUMP":
        data_path = f"data/PUMP/pump.npz"
    data = np.load(data_path)
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    
    # 创建窗口化数据集
    train_dataset = WindowedDataset(X_train, config.seq_len, config.stride)
    val_dataset = WindowedDataset(X_val, config.seq_len, config.stride)
    test_dataset = TestDataset(X_test, y_test, config.seq_len, config.stride)

    # 检查第一个样本
    # sample = train_dataset[0]
    # print(f"单个样本形状: {sample.shape}")  # 应该是[100,35]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: torch.stack(x, dim=0)
    )
    # # 验证第一个batch
    # first_batch = next(iter(train_loader))
    # print(f"第一个batch形状: {first_batch.shape} (应为[512,100,35])")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda x: torch.stack(x)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch])
        )
    )
    test_sample, test_label = next(iter(test_loader))
    
    return train_loader, val_loader, test_loader
