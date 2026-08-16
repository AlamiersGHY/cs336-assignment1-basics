import torch 
import os
import typing

def save_checkpoint(
    model:torch.nn.Module,   #模型权重
    optimizer:torch.optim.Optimizer,  #m v等与当前step和参数有强烈关系的参数
    iteration:int,  #step
    out:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]]
):
    """
    保存当前训练状态。
    """
    # 1.构建一个包含所有必要信息的字典
    checkpoint={
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration
    }
    
    # 2.使用torch.save将字典写入目标
    torch.save(checkpoint,out)
    
    
def load_checkpoint(
    src:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]],
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer
)->int:
    """
    从检查点恢复状态，并返回保存时的迭代次数。
    """
    # 1.加载字典
    checkpoint=torch.load(src,map_location='cpu') #设备重定向
    
    # 2.参数恢复
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 3.返回迭代次数
    return checkpoint['iteration']
    
    