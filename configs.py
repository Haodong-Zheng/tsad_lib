import torch

class Config:

    dataset = "SDC"  # SDC(超算数据集),SMAP,PSM
    model = "DPFAN" # DPFAN,TimesNet,AnomalyTransformer,COUTA,DIF,DeepSVDD,ModernTCN,NCAD,FITS,iTransformer,TranAD

    seed = 42  # 随机种子
    stride = 1
    num_workers = 4
    
    # 模型参数
    seq_len=100  # 窗口大小
    pred_len=0
    d_model=64  # 模型嵌入维度
    d_ff=128
    top_k=5
    num_kernels=6
    e_layers=2
    dropout=0.1
    
    # 训练参数
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    lr = 1e-4
    epochs = 10
    patience = 3  # 早停机制
    
    # 路径配置
    model_save_path = f"checkpoints/{model}_{dataset}.pth"

config = Config()
