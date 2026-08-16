from typing import Any

import torch
import torch.nn as nn
import math

# Embedding模块
class Embedding(nn.Module):
    def __init__(self,
                num_embedding:int,embedding_dim:int,
                device=None,dtype=None) :
        
        super().__init__()
        
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(
            torch.empty((num_embedding,embedding_dim),**factory_kwargs))  
        std=1.0
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std)
        
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        
        return self.weight[token_ids]
    
    
# Linear模块
class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        
        std=(2.0/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=3*std,b=-3*std)
        
    def forward(self,X):
        return X@self.weight.T
        
      
# softmax函数  [N,M,....K]   dim决定对哪一维进行归一化
def softmax(X:torch.Tensor,dim:int=-1)->torch.Tensor:
    # 保证数值稳定性减去最大值
    X_max=torch.max(X,dim=dim,keepdim=True).values
    X=X-X_max
    
    X_exp=torch.exp(X)
    sum_exp=torch.sum(X_exp,dim=dim,keepdim=True)
    return X_exp/sum_exp
          


# Attention模块
def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask:torch.Tensor=None
)->torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    
    d_k=Q.size(-1)
    
    # 计算Q*K^T
    scores=(Q@K.transpose(-2,-1))/math.sqrt(d_k)  
    #添加掩码
    if mask is not None :
        scores=scores.masked_fill(mask==False,float('-inf'))
    # 注意只对最后一维softmax
    return softmax(scores,dim=-1)@V

# RoPE旋转位置编码
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, context_length: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        context_length: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k
        
        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)
        
        # 2. 创建位置序列 [0, 1, ..., context_length - 1]
        t = torch.arange(context_length, device=device).float()  # 形状: (context_length,)
        
        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (context_length, d_k/2)
        freqs_matrix = torch.outer(t, freqs)
        
        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., context_length, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
    

# 因果多头自注意力
class CasusalSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,bias:bool=False,
                 context_length=None,theta=None,
                 device=None,dtype=None):
        super().__init__()
        assert d_model%num_heads == 0,"d_model必须能被num_heads整除"
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        
        # 1.定义Q K V的投影层（包含初始化矩阵）
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        # 三个不同的矩阵参数 各自独立
        
        # 2.定义输出层
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        
        # 3.实例化RoPE
        if theta is not None and context_length is not None:
            self.rope=RotaryPositionalEmbedding(theta.self.d_k,context_length,device=device)
        else:
            self.rope=None
            
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        b,s,d=x.shape
        
        # 1.做投影之后拆分头
        q=self.q_proj(x)  #[B,S,d_model]
        q=q.reshape(b,s,self.num_heads,self.d_k) #[B,S,H,d_k]
        q=q.permute(0,2,1,3)  #[B,H,S,d_k]
        
        k=((self.k_proj(x)).reshape(b,s,self.num_heads,self.d_k)).permute(0,2,1,3) 
        v=((self.v_proj(x)).reshape(b,s,self.num_heads,self.d_k)).permute(0,2,1,3) 

        
        # 3.应用RoPE
        if self.rope is not None:
            if token_positions is None:
                batch_dims=x.shape[:-2]
                token_positions=torch.arange(s,device=x.device).expand(*batch_dims,s)
                
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)
            
            
        # 4.生成下三角掩码矩阵
        mask=torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))
        
        # 5.SDPA
        attn_out=scaled_dot_product_attention(q,k,v,mask)
        
        # 6.合并后线性整合输出
        return self.output_proj(attn_out.transpose(1,2).reshape(b,s,d))
    
    
# SiLU激活函数
def silu(in_features:torch.Tensor)->torch.Tensor:
    return in_features * torch.sigmoid(in_features)
    
#SwiGLU层
class SwiGLU(nn.Module) :
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None) -> None:
        super().__init__()
        
        self.d_model=d_model
        self.d_ff=d_ff
        
        # 初始化两个升维线性层（无偏置）
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)
        # 初始化一个降维层
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 计算特征门禁  [B,S,d_model]->[B,S,d_ff]
        gate=silu(self.w1(x)) 
        # 计算特征值    [B,S,d_model]->[B,S,d_ff]
        signal=self.w3(x)
        # 计算逐元素乘积（前后张量形状不变） 并通过降维层输出[B,S,d_ff]->[B,S,d_model]
        return self.w2(gate*signal)
    
    
# LayerNorm模块
class LayerNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None) -> None:
        super().__init__()
        
        # 1.学习参数初始化  weight和bias
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.bias=nn.Parameter(torch.zeros(d_model,device=device,dtype=dtype))
        
        self.eps=eps
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 输入x:[B,S,d_model]
        in_dtype=x.dtype
        # 2.转换为float32确保计算均值和方差时的数值稳定性（防止溢出）
        x_float=x.to(torch.float32)
        
        # 3，计算均值
        mean=x_float.mean(dim=-1,keepdim=True)  #保持最后一个维度存在不压缩 使得后续可以广播
        
        # 4.计算方差
        var=x_float.var(dim=-1,keepdim=True,unbiased=False)
        
        # 5.归一化处理
        x_normed=(x_float-mean)/torch.sqrt(var+self.eps)
        
        # 6.应用可学习参数  同时转为原数据类型输出
        return (x_normed*self.weight+self.bias).to(in_dtype)
    
# RMSNorm模块
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None) -> None:
        super().__init__()
        # 1.初始化权重参数
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype)) #Shape[d_model]
        
        self.eps=eps
        
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_type=x.dtype
        x_float=x.to(torch.float32)  
        
        # 2.计算均方根
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
    
        # 3.归一化处理
        x_normed=x_float/rms
        
        # 4.添加可学习参数 进行类型转换后输出
        return (x_normed*self.weight).to(in_type)
        
        
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,context_length:int,
                 theta:float,device=None,dtype=None,
                 norm_type:str="rmsnorm",  #"rmsnorm"或"layernorm"
                 norm_mode:str="pre",   #"pre"或”post
                 ffn_type:str="swiglu"  #"swiglu"或"silu"
                 ) -> None:
        super().__init__() 
        # 保存一下几种内部模块和类型的选取
        self.norm_type=norm_type
        self.norm_mode=norm_mode
        self.ffn_type=ffn_type
        
        # 1.初始化Attention层
        self.attn=CasusalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            theta=theta,
            device=device,
            dtype=dtype)
        
        # 2.初始化两个norm层
        if norm_type == "rmsnorm":
            # 使用rmsnorm
            self.norm1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
            self.norm2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        elif norm_type == "layernorm":
            # 使用layernrom
            self.norm1=LayerNorm(d_model=d_model,device=device,dtype=dtype)
            self.norm2=LayerNorm(d_model=d_model,device=device,dtype=dtype)
        else :
            # 禁用norm层(消融实验做准备)
            self.norm1=nn.Identity()
            self.norm2=nn.Identity()
            
            
        # 3.初始化FFN层
        if ffn_type =='swiglu':
            self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        elif ffn_type =='silu':
            # 采用标准FFN：x->Linear->Silu->Linear->out
            #与SWiGlu采用不同的架构(其d_ff由最外层脚本指定)，两者所使用的参数数量不同，所以这里需要把外部的d_ff覆盖为4*d_model
            d_ff=4*d_model
            self.ffn=nn.Sequential(
                Linear(d_model,d_ff,device=device,dtype=dtype),
                nn.SiLU(),
                Linear(d_ff,d_model,device=device,dtype=dtype)
            )
        else:
            # 兜底分支
            raise ValueError(f"Unknown ffn_type:{ffn_type}")

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        # Pre-norm
        # 公式: x = x + Sublayer(Norm(x))
        if self.norm_mode =='pre':
            x=x+self.attn(self.norm1(x),token_positions=token_positions)
            x=x+self.ffn(self.norm2(x))
            
        # Post-norm
        # 公式: x = Norm(x + Sublayer(x))
        elif self.norm_mode=='post':
            x=self.norm1(x+self.attn(x,token_positions))
            x=self.norm2(x+self.ffn(x))
            
        return x  


# TransformerLM
class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,context_length:int,d_model:int,
                 num_layers:int,num_heads:int,d_ff:int,rope_theta:float,
                 device=None,dtype=None,
                #  实验参数
                 norm_type:str='rmsnorm',norm_mode:str='pre',ffn_type:str='swiglu'
                 ) -> None:
        super().__init__()
        self.context_length=context_length
        
        # 1.初始化Token Embedding层
        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)

        # 2.初始化Transformer Block Layers
        self.layers=nn.ModuleList([TransformerBlock(d_model,num_heads, d_ff, context_length,rope_theta,
                                     device=device,dtype=dtype,
                                     norm_mode=norm_mode,norm_type=norm_type,ffn_type=ffn_type)
                                   for _ in range(num_layers)]
            
        )
        
        # 3.初始化输出前的Norm层
        if norm_type=='rmsnorm':
            self.norm=RMSNorm(d_model,device=device,dtype=dtype)
        elif norm_type=='layernorm':
            self.norm=LayerNorm(d_model,device=device,dtype=dtype)
        else:
            # 禁用归一化
            self.norm=nn.Identity()
        
        # 4,初始化最后一个输出前的线性映射层
        self.out=Linear(d_model,vocab_size,device=device,dtype=dtype)
        
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        
        # 保存token_ids的形状[B,S]
        b,s=token_ids.shape
        
        # 准备token_positions
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)
        
        # 1.Embedding
        x=self.token_embeddings(token_ids)
        
        # 2.逐层通过Transformer block
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)  #带上运行时的token_positions
        
        # 3.归一化
        x=self.norm(x)
        
        # 4.线性输出层
        return self.out(x)
            
        

