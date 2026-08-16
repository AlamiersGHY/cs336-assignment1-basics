import torch,math
from collections.abc import Iterable



# 1.SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params,lr=1e-3) -> None:
        """
        params: 传入模型需要优化的参数（通常是 model.parameters()）。
        defaults: 一个字典，存储默认超参数（如学习率 lr）。
        调用 super().__init__ 后，PyTorch 会将参数组织在 self.param_groups 中
        """
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults={'lr':lr}
        super().__init__(params,defaults)
        
        
    def step(self):
        loss=None
        for group in self.param_groups:
            lr =group["lr"]
            # 可能有多个参数(组) 例如线性层有weight和bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 拿到参数和lr之后  获取当前参数的状态字典
                state=self.state[p]
                if len(state) == 0:
                    state['t'] = 0   #优化器初次初始化的时候还没有开始记录状态 现在的state是空字典 因而需要初始化t
                    
                t =state['t']
                grad=p.grad.data
                
                # 执行更新公式
                p.data-=lr/math.sqrt(t+1)*grad
                # 更新步数
                state['t']+=1
                
        return loss
    
def run_experiment(learning_rate):
    print(f"\n--- Testing LR ={learning_rate} ---")
    # 初始化权重
    weights=torch.nn.Parameter(5*torch.randn((10,10)))
    opt=SGD([weights],lr=learning_rate)
    
    for t in range(10):
        opt.zero_grad()
        # 计算Loss
        loss=(weights**2).mean()
        print(f"Iter {t}:Loss = {loss.item():.4f}")
        loss.backward()
        opt.step()
        
# 学习率测试
# lrs_to_test = [1e1,1e2,1e3]
# for lr in lrs_to_test:
#     run_experiment(lr)
    
"""
Test Result:
--- Testing LR =10.0 ---
Iter 0:Loss = 29.6385
Iter 1:Loss = 18.9687
Iter 2:Loss = 13.9829
Iter 3:Loss = 10.9401
Iter 4:Loss = 8.8615
Iter 5:Loss = 7.3472
Iter 6:Loss = 6.1964
Iter 7:Loss = 5.2950
Iter 8:Loss = 4.5726
Iter 9:Loss = 3.9833

--- Testing LR =100.0 ---
Iter 0:Loss = 22.1342
Iter 1:Loss = 22.1342
Iter 2:Loss = 3.7976
Iter 3:Loss = 0.0909
Iter 4:Loss = 0.0000
Iter 5:Loss = 0.0000
Iter 6:Loss = 0.0000
Iter 7:Loss = 0.0000
Iter 8:Loss = 0.0000
Iter 9:Loss = 0.0000

--- Testing LR =1000.0 ---
Iter 0:Loss = 24.1578
Iter 1:Loss = 8720.9648
Iter 2:Loss = 1506247.8750
Iter 3:Loss = 167553984.0000
Iter 4:Loss = 13571872768.0000
Iter 5:Loss = 856540577792.0000
Iter 6:Loss = 43971996811264.0000
Iter 7:Loss = 1891863300669440.0000
Iter 8:Loss = 69730043886043136.0000
Iter 9:Loss = 2239109113038503936.0000

"""

#2.AdamW
class AdamW(torch.optim.Optimizer):
    def __init__(self, params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01) -> None:
        # 1.基本参数检查
        if lr<0.0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if not 0.0<=betas[0]<1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[0]}")
        if not 0.0<=betas[1]<1.0:
            raise ValueError(f"Invalid beta parameter at index 1:{betas[1]}")
        if eps<0:
            raise ValueError(f"Invalid epslion value:{eps}")
        
        # 2.将超参数存入defaults字典
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_dacay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss=None
        
        for group in self.param_groups:
            beta1,beta2=group["betas"]
            eps=group['eps']
            lr=group['lr']
            wd=group['weight_decay']

            for p in group['params']:
                # 首先检查是否反向传播回梯度
                if p.grad is None:
                    continue
                
                # 获取梯度和状态表
                grad=p.grad
                state=self.state[p]
                
                # 3.状态初始化(第一次)
                if len(state)==0:
                    state['step']=0
                    # m一阶矩
                    state['exp_avg']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    # v二阶矩
                    state['exp_avg_sq']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    
                exp_avg,exp_avg_sq=state['exp_avg'],state['exp_avg_sq']
                state['step']+=1
                t=state['step']
                    
                    
                # 4.更新矩估计
                # m = beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                
                # 5，计算偏差矫正后的学习率
                bias_correction1=1-beta1**t
                bias_correction2=1-beta2**t
                step_size=lr*(math.sqrt(bias_correction2)/bias_correction1) 
                
                #6.更新参数
                denom=exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg,denom,value=-step_size)
            
                # 7.权重衰减项
                if wd!=0:
                    p.add_(p,alpha=-lr*wd)
                    
        return loss
    
    
#梯度裁剪
def clip_gradient_norm(parameters:Iterable[torch.nn.Parameter],max_norm:float):
    """
    实现梯度裁剪（Global Norm Clipping）。
    
    参数:
        parameters: 可迭代的参数列表（通常是 model.parameters()）
        max_norm: 允许的最大梯度的 L2 范数 (M)
    """
    # 1.过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 
    
    # 2.计算全局L2范数
    total_norm=0.0
    
    for p in params_with_grad:
        # 注意使用.detach()避免数值操作进入计算图
        param_norm=torch.norm(p.grad.detach(),p=2)  #计算当前p.grad的L2范数
        total_norm+=param_norm.item()**2
        
        
    total_norm=total_norm**0.5
    
    # 3.检查阈值i
    eps=1e-6
    if total_norm>max_norm:
        # 计算缩放因子
        clip_coef=max_norm/(total_norm+eps)
        
        # 4.原地修改每个餐宿的梯度
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)