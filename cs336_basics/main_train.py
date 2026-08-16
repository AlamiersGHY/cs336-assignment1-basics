import argparse
import os
from typing import Protocol, cast

import torch
import numpy as np
import wandb
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradient_norm
from cs336_basics.scheduler import get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.losses import cross_entropy


# 用 Protocol 描述 argparse.Namespace 的字段结构，让编辑器能对 args 提供补全和类型检查
# （Namespace 运行时是有属性的对象而非 dict，所以用 Protocol 而不是 TypedDict）
class TrainArgs(Protocol):
    batch_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    vocab_size: int
    norm_type: str
    norm_mode: str
    no_rope: bool
    ffn_type: str
    lr: float
    max_iters: int
    warmup_iters: int
    min_lr: float
    max_norm: float
    train_data_path: str
    valid_data_path: str
    out_dir: str
    device: str
    wandb_project: str
    run_name: str | None
    seed:int


def main():
    parser=argparse.ArgumentParser()
    # --- 模型超参数 ---
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--context_length",type=int,default=256)
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_layers",type=int,default=4)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--d_ff",type=int,default=2048)
    parser.add_argument("--vocab_size",type=int,default=10000)
    
    # --- 实验参数 ---
    # 1.Norm_type
    parser.add_argument("--norm_type",type=str,default='rmsnorm',choices=['rmsnorm','layernorm','none'],help="norm_type")
    # 2.Norm_mode
    parser.add_argument("--norm_mode",type=str,default='pre',choices=['pre','post'],help='Normalization placement')
    # 3.是否rope
    parser.add_argument("--no_rope",action="store_true",help="Disable Rotray Position Embedding")
    # 4.FFN_type
    parser.add_argument("--ffn_type",type=str,default="swiglu",choices=['swiglu','silu'])
    
    # --- 优化器参数 ---
    parser.add_argument("--lr",type=float,default=6e-4) 
    parser.add_argument("--max_iters",type=int,default=10000)
    parser.add_argument("--warmup_iters",type=int,default=1000)    
    parser.add_argument("--min_lr",type=float,default=6e-5)
    parser.add_argument("--max_norm",type=float,default=1.0)
    
    # 路径和系统
    parser.add_argument("--train_data_path",type=str,required=True)
    parser.add_argument("--valid_data_path",type=str,required=True)
    parser.add_argument("--out_dir",type=str,default="out")
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Wandb设置
    parser.add_argument("--wandb_project",type=str,default="cs336-pretraining")
    parser.add_argument("--run_name",type=str,default=None,help="Wandb 实验名称")
    
    # Seed随机种子设置
    parser.add_argument("--seed",type=int,default=42)
    
    
    # 读取命令行参数并转存（cast 只是静态类型标注，运行时不产生任何开销）
    args = cast(TrainArgs, parser.parse_args())
    # 固定seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.out_dir,exist_ok=True)
    
    # 1.加载数据(使用memmap)
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")
    
    # np.memap 延迟加载到内存 
    train_data=np.memmap(args.train_data_path,dtype=np.uint16,mode='r')
    val_data=np.memmap(args.valid_data_path,dtype=np.uint16,mode='r')
    print(f"训练集大小: {len(train_data)} tokens")
    print(f"测试集大小: {len(val_data)} tokens")    
    
    # 2.消融实验
    actual_rope_theata=None if args.no_rope else 10000.0
    
    # 3.初始化模型
    model=TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theata,
        device=args.device,
        # 实验参数
        norm_type=args.norm_type,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)
    
    print(f"Model Config :Norm_type={args.norm_type},Norm_mode={args.norm_mode},ffn_type={args.ffn_type}")
    
    # --- 设备状态自检 ---
    param_device=next(model.parameters()).device
    print(f"模型实际设备: {param_device}")

    
    # 4.初始化优化器
    optimizer =AdamW(model.parameters(),lr=args.lr,weight_decay=0.1)
    
    # 5.检查点恢复逻辑（首次或者当本次实验中还没有checkpoint的时候正常进入实验 但是当实验中途中断 可以从此处再自动读取checkpoint重新开始实验）
    start_iter=0
    ckpt_path=os.path.join(args.out_dir,"ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter=load_checkpoint(ckpt_path,model,optimizer)
        print(f"Resuming from iteration {start_iter}")
        
    # 6.初始化WandB
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    # 7.主训练循环
    for it in range(start_iter,args.max_iters):
        # a.更新学习率
        lr=get_lr_cosine_schedule(it,args.lr,args.min_lr,args.warmup_iters,args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
            
        # b.训练
        model.train()
        
        # 准备数据batch
        x,y=get_batch(train_data,args.batch_size,args.context_length,device=args.device)
        
        # 前向传播
        logits=model(x)
        
        # 计算Loss
        loss=cross_entropy(logits,y)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        clip_gradient_norm(model.parameters(),max_norm=args.max_norm)
        
        # 更新参数
        optimizer.step()
        
        # c.验证与日志记录
        if it%100==0 or it==args.max_iters-1:
            model.eval()
            # 停止梯度计算
            with torch.no_grad():
                vx,vy=get_batch(val_data,args.batch_size,args.context_length,device=args.device)
                v_logits=model(vx)
                v_loss=cross_entropy(v_logits,vy)
                print(f"Iter:{it},train_loss:{loss.item():.4f},val_loss:{v_loss.item():.4f}")
                wandb.log({
                    "train/loss":loss.item(),
                    "val/loss":v_loss.item(),
                    "lr":lr,
                    "iter":it+1
                })
        # d.保存检查点（每1000步存一次）
        if it%1000==0 and it >0:
            save_checkpoint(model,optimizer,it,out=ckpt_path)
            
    # 8.训练结束保存最终模型
    save_checkpoint(model,optimizer,args.max_iters,out=os.path.join(args.out_dir,"ckpt_final.pt"))
    wandb.finish()
    
if __name__=="__main__":
    main()
    
    
                
        
    