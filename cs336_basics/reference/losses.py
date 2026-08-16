import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
    计算交叉熵损失。
    
    参数:
        logits: 预测的分数，形状为 (..., vocab_size)
        targets: 目标 ID，形状为 (...)
        
    返回:
        平均损失标量。
    """
    # 1.提取维度信息
    vocab_size=logits.size(-1)
    
    # 2.取max  保留维度信息，后续需要广播做减法  [B,S,1]
    m=torch.max(logits,dim=-1,keepdim=True).values
    
    # 3.提取目标位置的Logits(o_y)  [B,S]
    target_logits=torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)    
    
    # 4.计算logSumExp项  [B,S]
    shifted_logits=logits-m  #[B,S,vocab_size]
    log_sum_exp=m.squeeze(-1)+torch.log(torch.sum(torch.exp(shifted_logits),dim=-1))
    
    # 5.计算单个token的loss [B,S]
    loss=log_sum_exp-target_logits
    
    # 6.返回整个batch的标量（平均值)
    return loss.mean()