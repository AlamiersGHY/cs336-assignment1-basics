from typing import Any

import torch,math
from torch import nn
from einops import rearrange



# Embedding模块
# 内部维护一个token映射词表[V,D]
# 输入：[B,S]->输出[B,S,D]  

class Embedding(nn.Module):
    def __init__(self,V:int,D:int,device=None,dtype=None):
        
        super().__init__()
        # 初始化一块空间
        self.weight=nn.Parameter(torch.empty((V,D),device=device,dtype=dtype))
        # 初始化值
        std=1.0
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std)
        
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]  #批量查找 
    
    
# Linear模块  
# 内部维护一个weight形状为[out_features,in_features]  与社区实践一致采用行的形式
# 输入X[...,in_features]@weight.T[in_features,out_features]->输出[...,out_features]
class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        
        # 初始化
        self.weight=nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        std=(2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std)
            
    def forward(self,X:torch.Tensor)->torch.Tensor:
        return X@self.weight.T
        
#softmax函数 可以对任意形状的tensor进行softmax同时可以指定维度 默认对最后一维softmax
def softmax(x:torch.Tensor,dim:int=-1)->torch.Tensor:
    # 减去每一行的max维持稳定性
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x=x-x_max
    
    x_exp=torch.exp(x)
    x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    return x_exp/x_sum
        


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
    
    
# Attentionmo模块（缩放点击注意力）
# 输入：Q,K，V以及一个mask（表示要注意和保留哪些位置）
# 输出：经过点积注意力softmax等过后的Attention矩阵
# 形状说明：Q[...,n,d_k]  K[...,m,d_k]  V[...,m,d_v]  mask[...,n,m]的布尔值tensor
def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor|None=None)->torch.Tensor:
    d_k=Q.size(-1)  #取形状的最后一维的数值
    # 计算Q*K^T  (K.transpose(-2,-1)只转置最后两维, 而K.T会反转全部维度)
    prod=(Q@K.transpose(-2,-1))/math.sqrt(d_k)
    # 加上掩码 (mask=True的位置保留, mask=False的位置填-inf)
    if mask is not None:
        prod=prod.masked_fill(mask==False,float('-inf'))
    # 计算softmax并与V做点积
    return softmax(prod,dim=-1)@V  #输出注意力矩阵[...,n,d_v]
    
    
    
# 因果多头自注意力模块
# 输入：一个[B,S,D]等的多维矩阵（可能有多种batch形状）
# 首先经过线性层的权重初始化计算出Q K V，然后需要对Q K应用位置编码，随后分头（维度变换），每个头进入对应的SDPA子层（有掩码），最后拼接，然后经过一个线性层整合输出
# 输出：一个注意力权重矩阵[B,S,D]
class CasusalSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,
                 device=None,dtype=None,
                 context_length=None,theta=None) -> None:
        
        super().__init__()
        assert d_model%num_heads == 0,"num_heads必须能被d_model整除"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads #计算出子层的特征维度
        
        # 1.初始化线性层（具有独立的权重参数）
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        
        # 2.准备最后的整合输出线性层
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        
        # 3.准备位置编码
        if theta is not None and context_length is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,context_length,device=device)
        else:
            self.rope=None
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        
        #保存一下x的形状 便于后续恢复
        b,s,d=x.shape  #[B,S,D]
    
        # 1.计算Q K V  [B,S,D]->[B,S,D]  同时先进行分头处理
        q=rearrange(self.q_proj(x),"... s (h d_k)->... h s d_k",h=self.num_heads)
        k=rearrange(self.k_proj(x),"... s (h d_k)->... h s d_k",h=self.num_heads)
        v=rearrange(self.v_proj(x),"... s (h d_k)->... h s d_k",h=self.num_heads)
 
        # 2.添加位置编码
        if self.rope is not None:
            # 检查token_positions是否传入
            if token_positions is None:
                # 取X的除去后两个维度以外的batch维度(防御)
                batch_dims=x.shape[:-2]
                token_positions=torch.arange(end=s,device=x.device).expand(*batch_dims,s)  #expand之前是[0,1,2...]的一维[S] 需要的是[B,S]对于不同的batcha维度不同的seq均有对应的token_positions
            # 有rope有token_positions
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)    
            
        # 3.准备掩码
        mask=torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
        
        # 4.进入SPDA
        output=scaled_dot_product_attention(q,k,v,mask=mask)  
        #[B,H,S,d_k]->[B,S,D]
        output=rearrange(output,"... h s d-> ... s (h d)")  #此时是合并已经知道张量的维度 不需要再给定h
        
        # 5.整合输出
        return self.output_proj(output)

        

# SiLU激活函数
def silu(x:torch.Tensor)->torch.Tensor:
    return x*torch.sigmoid(x) 

# SwiGLU前馈全连接层
# 输入 ：一个以d_model为特征维度的张量x[B,S,d_model]
# 经过SwiGLU,分别计算门禁和信息值（升维后的），逐元素相乘后再经过一个线性层降维输出
# 输出：一个与输入相同形状的tensor
# 要求参数d_model,d_ff（指定升维维度）
class SwiGLU(nn.Module):
    def __init__(self, d_model:int,d_ff:int,device=None,dtype=None) -> None:
        super().__init__()
        
        self.d_model=d_model
        self.d_ff=d_ff
        
        # 初始化两个升维线性层
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)
        # 初始化一个降维输出层
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 计算门禁gate
        gate=silu(self.w1(x))
        # 计算特征值
        signal=self.w3(x)
        
        return self.w2(gate*signal)
    
    
# RMSNorm模块
# 输入：一个等待归一化处理的矩阵x[B,S,d_model]
# 经过RMSNorm模块的归一化处理后方差变为1，使得整体的数据前后传输稳定，同时内部有一个可学习的参数矩阵（可以针对每个类型的token进行特定的适当的调整）
# 参数约定：weight一维[d_model]随后自行广播到所有的[B,S]  eps全局共享的小型调整项
# 输出：一个归一化后的RMSNorm(x) [B,S,d_model]
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None) -> None:
        super().__init__()
        
        # 1.初始化参数矩阵
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    
        self.eps=eps
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 数据类型处理（后续要进行较为大的平方和处理）
        in_dtype=x.dtype
        x_float=x.to(torch.float32)
        
        # 2.计算方均根
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
        
        # 3.归一化处理
        x_normed=x_float/rms
        
        # 4.添加可学习参数恢复类型并输出(广播计算)
        return (x_normed*self.weight).to(in_dtype)
     

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




#Trandformer Block
# 输入x 输出x [B,S,d_model]
# 模式约定（可扩展、可消融） norm_type:'rmsnorm'\'layernorm'\none  norm_mode:'pre'\'post'  ffn_type:'swiglu'\'silu'(传统ffn)
# 其他参数约定：d_model、d_ff、num_heads、context_length、theta、device、dtype
# cs336标准数据流：x->norm->Attention->add->norm->ffn->add->x
class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,num_heads:int,context_length:int,theta:float,
                 norm_type:str='rmsnorm',norm_mode:str='pre',ffn_type:str='swiglu',
                 device=None,dtype=None) -> None:
        super().__init__()
        
        # 保存模式参数
        self.norm_type=norm_type
        self.norm_mode=norm_mode
        self.ffn_type=ffn_type
        
        # 1.初始化注意力层
        self.attn=CasusalSelfAttention(d_model=d_model,num_heads=num_heads,device=device,dtype=dtype,context_length=context_length,theta=theta)
        
        # 2.初始化Norm层(默认RMSNorm)
        if norm_type=='rmsnorm':
            self.norm1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
            self.norm2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        elif norm_type =='layernorm':
            self.norm1=LayerNorm(d_model=d_model,device=device,dtype=dtype)
            self.norm2=LayerNorm(d_model=d_model,device=device,dtype=dtype)
        elif norm_type =='none': #消融实验 不进行归一化
            self.norm1=nn.Identity()
            self.norm2=nn.Identity()
        else: 
            raise ValueError(f"Unknown norm_type:{norm_type}")
        
        # 3.初始化ffn层(默认SWiglu,d_ff交给外部确定)
        if ffn_type=='swiglu':
            self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        elif ffn_type=='silu':
            # Linear->silu->Linear
            d_ff=4*d_model  #确保参数量一致 否则该处的参数会少
            self.ffn=nn.Sequential(
                Linear(d_model,d_ff,device=device,dtype=dtype),
                nn.SiLU(),
                Linear(d_ff,d_model,device=device,dtype=dtype)
            )
        else: #前反馈层作为表达必不可缺
            raise ValueError(f"Unknown ffn_type:{ffn_type}")
                
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        if self.norm_mode == 'pre':
            # pre-norm
            x=x+self.attn(self.norm1(x),token_positions=token_positions)
            x=x+self.ffn(self.norm2(x))
        elif self.norm_mode =='post':
            # post-norm
            x=self.norm1(x+self.attn(x,token_positions=token_positions))
            x=self.norm2(x+self.ffn(x))
        else:  #防御代码
            raise ValueError(f"Unknown norm_mode:{self.norm_mode}")

        return x
    


# TransformerLM组装模块
# 将之前的block堆叠起来 同时补上前后的embedding和线性输出层
# 模块规划：x->Embedding->n*transformer block->norm->Linear->x
# 参数约定：必需:vocab_size  num_layers、d_model、d_ff、context_length、num_heads、rope_theta   可选:device、dtype   实验参数:norm_type、norm_mode、ffn_type

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,num_layers:int,d_model:int,d_ff:int,context_length:int,num_heads:int,rope_theta:float,
                 device=None,dtype=None,
                 norm_type:str='rmsnorm',  #rmsnorm\layernorm\none
                 norm_mode:str='pre',      #pre\post
                 ffn_type:str='swiglu'     #swiglu\silu
                 ) -> None:
        super().__init__()        
    
        # 保存参数
        self.context_length=context_length
        self.d_model=d_model
        
        
        # 1.初始化Token Embedding层
        self.token_embedding=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        
        # 2.初始化Transformer Block
        self.layers=nn.ModuleList([
            TransformerBlock(d_model,d_ff,num_heads,context_length,rope_theta,
                             norm_type,norm_mode,ffn_type,
                             device=device,dtype=dtype)
            for _ in range(num_layers)
        ])
        
        # 3.初始化norm层
        if norm_type =='rmsnorm':
            self.norm=RMSNorm(d_model,device=device,dtype=dtype)
        elif norm_type == 'layernorm':
            self.norm=LayerNorm(d_model,device=device,dtype=dtype)
        elif norm_type == 'none':
            self.norm=nn.Identity()  #禁用Norm
        else:
            raise ValueError(f"Unknown norm_type:{norm_type}")
        
        # 4.初始化最后的线性输出层
        self.out=Linear(d_model,vocab_size,device=device,dtype=dtype)
        
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        # 保存x的形状
        b,s=token_ids.shape
        
        # 准备token_positions
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)
        
        # 1.Embedding
        x=self.token_embedding(token_ids)

        # 2.进入blocks
        for layer in self.layers:
            x=layer(x,token_positions)
        
        # 3.Norm
        x=self.norm(x)
        
        # 4.线性输出
        return self.out(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:torch.Tensor,
        max_new_tokens:int,
        eos_token_id:int|None=None,
        temperature:float=1.0,
        top_p:float=1.0
    )->torch.Tensor:
        """
        从模型生成文本 ID 序列。
        
        参数: 
            prompt_ids: 提示词 ID (Batch, Seq_len)
            max_new_tokens: 最多生成的词数
            eos_token_id: 停止生成的 Token ID (如 <|endoftext|>)
            temperature: 温度系数 (越高越随机，越低越确定)
            top_p: 核采样阈值
        """ 
        # 设置为评估模式（永久改变 后续需要主动改为train)
        self.eval()
        
        # 拷贝原始数据
        generated=prompt_ids.clone()
        
        # 限定最大串行生成次数
        for _ in range(max_new_tokens):
            # 1.裁剪输入(只能处理conext_length的文本)
            idx_cond=generated[:,-self.context_length:]   #从[B,S]两个维度上切片 都取最后contect_length
            
            # 2.前向传播
            logits=self.forward(idx_cond)  #[B,S,V]
            logits=logits[:,-1,:]  #[B,V]只关心Seq维度的最后一个logits
            
            # 3.应用温度 (当t>1时 温度的数值越大 logits越平均 随机性越大；当t<1时，温度的数值越小，logits越发散，随机性越小)
            if temperature != 1.0:
                logits=logits/(temperature+1e-8)
                
            # 4.应用Top-p 过滤logits
            if top_p<1.0:
                logits=self._top_p_filter(logits, top_p)
            # 别过滤的logit被设置为-inf
            
            
            # 5.归一化采样
            probs=softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,num_samples=1)  #用于tensor采样的函数  [B,1]
            
            # 6.拼接新词
            generated=torch.cat((generated,next_token),dim=-1)
            
            #7.检测eos
            if eos_token_id is not None and (next_token ==eos_token_id).all():
                break
            
        return generated
            
            
            
            
    # 内部的过滤方法
    def _top_p_filter(self,logits:torch.Tensor,p:float)->torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        sorted_logits,sorted_indices=torch.sort(logits,descending=True,dim=-1)
        
        # 计算累积概率分布
        cumulative_probs=torch.cumsum(softmax(sorted_logits,dim=-1),dim=-1)
        
        # 创建掩码
        sorted_indices_to_remove=cumulative_probs>p
        
        # 调整掩码位置 使得第一位和第一个超过p的位置均为F
        sorted_indices_to_remove[...,1:]=sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0]=False
        
        # 将被移除的token设为-inf
        indices_to_move=sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
        logits=logits.masked_fill(indices_to_move,float('-inf'))
        
        return logits       


    

# if __name__=='__main__':
   