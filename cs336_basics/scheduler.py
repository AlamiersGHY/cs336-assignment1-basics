import math


# 余弦退火学习率调度器
# 分三个阶段：1.初期的warmuo线性阶段，增长到max_lr 2.余弦退火阶段，按照余弦规律逐渐降低学习率max->min 3.退火后阶段，学习率稳定在min_lr
def get_lr_cosine_schedule(
    it:int,
    max_learning_rate:float,
    min_learning_rate:float,
    warmup_iters:int,
    cosin_cycle_iters:int   
)->float:
    
    # 1.判断warmup阶段
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    
    # 2.判断退火后阶段
    if it>cosin_cycle_iters:
        return min_learning_rate
    
    # 3.计算余弦退火阶段学习率
    ratio=0.5*(1+math.cos(math.pi*(it-warmup_iters)/(cosin_cycle_iters-warmup_iters)))
    return min_learning_rate+ratio*(max_learning_rate-min_learning_rate)