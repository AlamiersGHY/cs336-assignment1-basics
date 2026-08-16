import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    context_length:int,
    device:str
)->tuple[torch.Tensor,torch.Tensor]:
    """
    从 Numpy 数组数据集中随机采样一个批次。
    
    参数:
        dataset: 1D Numpy 数组 (Token IDs)
        batch_size: 批大小 (B)
        context_length: 上下文长度 (m)
        device: 设备字符串 ('cpu', 'cuda', 'mps')
    """
    
    # 1.确定合法的最大起点索引
    n=len(dataset)
    max_idx=n-context_length-1
    
    # 2.随机产生batch_size个x的起始位置
    ix=torch.randint(0,max_idx+1,(batch_size,))
    
    # 3.根据随机索引提取对应的输入x和目标y
    # x: dataset[i : i+m]
    # y: dataset[i+1 : i+m+1]
    x_stack=[dataset[i:i+context_length] for i in ix]
    y_stack=[dataset[i+1:i+context_length+1] for i in ix]
    # 现在的x和y均为堆叠起来的numpy数组
    
    # 4.转换为Tensor并移动到指定设备
    x=torch.from_numpy(np.array(x_stack)).to(device).long()
    y=torch.from_numpy(np.array(y_stack)).to(device).long()
    
    return x,y #已经是Tensor