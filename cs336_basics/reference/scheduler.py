import math

def get_lr_cosine_schedule(
    it:int, #纯函数映射
    max_learing_rate:float,
    min_learning_rate:float,
    warm_up_iters:int,
    cosin_cycle_iters:int
)->float:
    """
    计算带预热的余弦退火学习率。
    
    it: 当前迭代次数 (t)
    max_learning_rate: 最大学习率 (alpha_max)
    min_learning_rate: 最小学习率 (alpha_min)
    warmup_iters: 预热步数 (T_w)
    cosine_cycle_iters: 总退火步数 (T_c)
    """
    
    # 1.判断预热阶段
    if it<warm_up_iters:
        return max_learing_rate*it/warm_up_iters
    
    # 2.判断退火后阶段
    if it>cosin_cycle_iters:
        return min_learning_rate
    
    # 3.处在余弦退火阶段
    # 计算纵坐标换算比例
    ratio=0.5*(1+math.cos(math.pi*(it-warm_up_iters)/(cosin_cycle_iters-warm_up_iters)))
    # 返回退火阶段学习率
    return min_learning_rate+ratio*(max_learing_rate-min_learning_rate)