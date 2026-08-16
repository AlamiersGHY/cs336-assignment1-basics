import torch 
import numpy as np
import numpy.typing as npt

# get_batch模块
# 传入dataset:npt.DNArray 内部包含该数据集中所有的经过分词和BPE之后的所有的token_ids,为一个一个的tokne
# 传入batch_size:规定要筛出的一个batch的大小
# 传入context_length:指定sequence长度
# 传入device:指定的数据存储设备
# 输出一个元组(x,y)  为经过筛选后的训练和目标Tensor,相互对应
# 模块职责:从传入的dataset(numpy数组)中随机取出batch_size组的训练(目标)token,每组(连续)token的长度为context_length,最后将整理完成的训练x和目标y的Tensor输出
def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    context_length:int,
    device:str
)->tuple[torch.Tensor,torch.Tensor]:
    
    n=len(dataset)
    #1.计算x下标最大能够选取的数值
    max_idx= n-context_length-1
    
    # 2.生成batch_size个随机的整数索引起点
    ix=torch.randint(0,max_idx+1,(batch_size,))  #Tensor [B]
    
    # 3.计算切片并堆叠
    x_stack=[dataset[i:i+context_length] for i in ix]
    y_stack=[dataset[i+1:i+1+context_length] for i in ix]
    # List[nparray[context_length]]
    
    # 4.将x和y转换为tensor并输出
    x=torch.from_numpy(np.array(x_stack)).to(device).long()
    y=torch.from_numpy(np.array(y_stack)).to(device).long()
    # Tensor[B,S] dtype=int64
    return x,y
    