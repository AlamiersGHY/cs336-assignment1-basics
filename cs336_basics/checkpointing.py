import torch 
import os
import typing

#保存checkpoint
# 保存那些参数? 模型权重,优化器参数,迭代步数
# 保存到哪里? out输出文件路径
def save_checkpoint(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration:int,
    out:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]]
):
    """
    保存当前训练状态。
    """
    # 1.将权重参数都整合到一个字典中
    check_point={
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration
    }
    
    # 2.使用torch.save()保存到out
    torch.save(check_point,out)
    
    
# 加载checkpoint
# 是save_checkpoint的逆操作
# 将模型参数等从文件中再加载回新的模型 同时返回iteration
def load_checkpoint(
    src:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]],
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer
)->int:
    """
    从检查点恢复状态，并返回保存时的迭代次数。
    """
    # 1.将文件中的数据加载回来
    check_point=torch.load(src,map_location='cpu')
    
    # 2.恢复参数
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    
    # 3.返回iteration
    return check_point['iteration']