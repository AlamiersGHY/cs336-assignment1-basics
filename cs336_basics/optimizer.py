import torch,math
from torch import nn
from collections.abc import Iterable

# 带有学习率衰减的SGD
# 输入：一个模型参数列表，学习率lr
# opt内部对模型的参数和对应的学习率等超参数进行打包，同时根据反向传播获得的梯度进行参数更新
# 输出：返回一个默认的无效loss
class SGD(torch.optim.Optimizer):
    def __init__(self, params,lr=1e-3) -> None:
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults={"lr":lr}
        super().__init__(params,defaults)
        
    def step(self):
        loss=None
        for group in self.param_groups:
            lr=group['lr']
            for p in group['params']:
                # 检查是否有grad
                if p.grad is None:
                    continue
                
                
                state=self.state[p]
                # 检查是否是第一次
                if len(state) ==0:
                    state['t']=0
                
                # 准备更新所需参数
                t=state['t']
                grad=p.grad.data
                # 执行更新
                p.data-=lr/math.sqrt(t+1)*grad
                
                # 更新步数
                state['t']+=1
        
        return loss
    
    
# AdamW优化器
# 参数约定：learning rate、beta1、beta2、eps、weight_dacay
# AdamW首先增加了m和v的动量机制（需要对应的参数），同时其动量值根据具体的梯度和步数会进行调整，m控制梯度更新的方向，v控制梯度更新的大小（使得更新幅度可控）
# ，同时为了使得更新的初始程度一致带有偏执纠正项，在具体实现中为了减少对tensor的计算，首先和lr进行标量估计合并计算，而后在随其与tensor计算
# 最后还有一个梯度衰减项(为了使得参数的值均不会过大)
class AdamW(torch.optim.Optimizer):
    def __init__(self, params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01) -> None:
        beta1,beta2=betas
        # 检查各个参数有效性
        if lr<=0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if not (0.0< beta1 <1 and 0.0<beta2<1):
            raise ValueError(f"Invalid betas:{beta1,beta2}")
        if eps<=0:
            raise ValueError(f"Invalid eps:{eps}")    
        
        # 组织defauts超参数
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        
        loss=None
        
        for group in self.param_groups:
            lr=group['lr']
            betas=group['betas']
            eps=group['eps']
            wd=group['weight_decay']
            
            beta1=betas[0]
            beta2=betas[1]
            
            for p in group['params']:
                # 检查梯度
                if p.grad is None:
                    continue
                
                # 初始化m,v以及step
                state=self.state[p]
                if len(state)==0:
                    # 初始化step
                    state['step']=0
                    # 初始化一阶矩
                    state['m']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    state['v']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    
                # 获取当前step标量 便于后续计算
                state['step']+=1
                t=state['step']
                
                # 准备m v 和grad
                m=state['m']
                v=state['v']
                grad=p.grad
                
                #更新m v 
                m.mul_(beta1).add_(grad,alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                
                # 计算偏差修正
                bias_correction1=1-beta1**t
                bias_correction2=1-beta2**t
                step_size=lr*math.sqrt(bias_correction2)/bias_correction1  
                
                #开始执行更新计算
                p-=step_size*m/(torch.sqrt(v)+eps)
                
                # 添加权重衰减
                p-=lr*wd*p

                
    
        return loss    
                
        

# 全局梯度裁剪
# 当学习过程中有些梯度过大还不进行处理可能会导致后续的梯度爆炸或者"步子"过大的问题
# 所以需要定期把模型所有参数的梯度抽出来检查一下他们的全局L2范数 看是否大于限定的最大值 如果大于则整体乘上一个缩放因子(保持梯度更新的方向不变)
# 参数约定:parameters传入模型所有的参数(是一个迭代器)可以逐个迭代出所有的参数 max_norm为限定的最大阈值
def clip_gradient_norm(parameters:Iterable[nn.Parameter],max_norm:float):
    # 1.筛选出有梯度的参数  同时整理成列表形式
    params_with_grad=[p for p in parameters if p.grad is not None]
    # 如果都没有梯度则不需要继续计算
    if not len(params_with_grad):
        return 
    
    # 2.计算单个参数的梯度的L2范数并累加
    total_norm=0.0  #标量
    for p in params_with_grad:
        p_norm=torch.norm(p.grad.detach(),p=2) #使用grad的时候从计算图摘取出来   返回Tensor
        total_norm+=p_norm.item()**2
        
    total_norm=total_norm**0.5
    
    # 3.检查是否大于阈值
    eps=1e-6
    if total_norm>max_norm:
        # 4.进行等比缩放
        ratio=max_norm/(total_norm+eps)
        for p in params_with_grad:
            p.grad.detach().mul_(ratio)  #对于每个参数的梯度抽取出来原地乘以ratio缩放系数
        
    
