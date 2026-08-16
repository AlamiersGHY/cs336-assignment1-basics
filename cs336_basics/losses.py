import torch


# Cross_entropy损失函数模块
# 输入：一个logits Tensor[B,S,vocab_size]表示经过模型的线性输出层之后的logits分数，一个需要拟合的目标next token Tensor[B,S]
# 公式：...见讲义
# 输出:loss.mean()的一维标量

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    # 保存参数信息
    vocab_size=logits.shape[-1]
    
    # 1.准备max 保留维度  max等函数一定一定注意.values()!!!
    m=torch.max(logits,dim=-1,keepdim=True).values  #[B,S,1]|[..,1]
    
    # 2.计算LosSumExp项
    log_sum_exp = m.squeeze(-1)+torch.log(torch.sum(torch.exp(logits-m),dim=-1)) #[B,S]|[...]
    
    # 3.准备o_y
    o_y=torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)  #[B,S] 
    
    # 4.计算loss
    loss=log_sum_exp-o_y #[B,S]-[B,S]=[B,S]
    
    # 5，返回loss标量（平均值）
    return loss.mean()