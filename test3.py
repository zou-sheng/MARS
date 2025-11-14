from bayes_opt import BayesianOptimization
import numpy as np

# --------------------------
# 1. 定义参数候选列表（与随机搜索一致）
# --------------------------
param_candidates = {
    'x1': [500, 800, 1000, 1500, 2000],
    'x2': [8, 16, 24, 32],
    'x3': [8, 16, 24, 32],
    'x4': [0.1, 0.5, 1.0, 1.5, 2.0]
}
param_order = ['x1', 'x2', 'x3', 'x4']

# 为贝叶斯优化创建「连续伪区间」（候选值的min→max）
param_bounds_bo = {
    param: (min(candidates), max(candidates))
    for param, candidates in param_candidates.items()
}

# --------------------------
# 2. 辅助函数：将连续值映射到最近的候选值
# --------------------------
def map_to_candidates(value, candidates):
    """将任意值映射到候选列表中最接近的值"""
    candidates = np.array(candidates)
    idx = np.argmin(np.abs(candidates - value))
    return candidates[idx]

# --------------------------
# 3. 目标函数（适配候选列表：先映射再计算损失）
# --------------------------
def bo_loss_function(x1, x2, x3, x4):
    # 步骤1：将贝叶斯优化采样的连续值映射到候选值
    x1 = map_to_candidates(x1, param_candidates['x1'])
    x2 = map_to_candidates(x2, param_candidates['x2'])
    x3 = map_to_candidates(x3, param_candidates['x3'])
    x4 = map_to_candidates(x4, param_candidates['x4'])
    
    # 步骤2：计算损失（与随机搜索一致）
    target = 10000
    output = x1 * x2 * x3 * x4
    return abs(output - target)

# --------------------------
# 4. 贝叶斯优化核心逻辑（适配候选列表）
# --------------------------
def bayesian_optimization_candidates(
    param_bounds_bo,     # 连续伪区间
    param_candidates,    # 真实候选列表
    param_order,         # 参数顺序
    loss_func,           # 目标函数
    n_init=30,           # 初始随机采样次数
    n_iter=100,          # 迭代优化次数
    random_seed=42
):
    # 初始化优化器
    optimizer = BayesianOptimization(
        f=loss_func,
        pbounds=param_bounds_bo,
        random_state=random_seed,
        verbose=1  # 精简日志
    )
    
    try:
        # 执行优化
        optimizer.maximize(
            init_points=n_init,
            n_iter=n_iter,
            acquisition_function="ei"  # 期望改进算法
        )
    except Exception as e:
        print(f"优化过程中出现异常：{e}")
        if not optimizer.res:
            raise ValueError("未获取到采样结果，优化失败")
    
    # 提取最优结果（遍历所有采样，找到损失最小的）
    if not optimizer.res:
        raise ValueError("无采样数据，无法获取最优解")
    
    # 遍历所有采样结果，筛选损失最小的
    best_idx = np.argmin([res["target"] for res in optimizer.res])
    best_res = optimizer.res[best_idx]
    best_params_raw = best_res["params"]
    
    # 步骤：将最优原始参数映射到候选值（确保是合法候选）
    best_params = [
        map_to_candidates(best_params_raw[param], param_candidates[param])
        for param in param_order
    ]
    best_loss = best_res["target"]
    
    # 转换为统一格式（x1/x2/x3为整数，x4保留1位小数）
    best_params = [
        int(best_params[0]),
        int(best_params[1]),
        int(best_params[2]),
        float(best_params[3])
    ]
    
    return best_params, best_loss

# --------------------------
# 5. 执行贝叶斯优化
# --------------------------
if __name__ == "__main__":
    print("=== 候选列表版贝叶斯优化开始 ===")
    try:
        best_params, best_loss = bayesian_optimization_candidates(
            param_bounds_bo=param_bounds_bo,
            param_candidates=param_candidates,
            param_order=param_order,
            loss_func=bo_loss_function,
            n_init=30,
            n_iter=100
        )
        
        print("\n=== 贝叶斯优化结果 ===")
        print(f"最优参数 [x1, x2, x3, x4]：{best_params}")
        print(f"最优损失：{best_loss:.2f}")
        print(f"对应输出 x1*x2*x3*x4：{np.prod(best_params):.2f}（目标：10000）")
    except Exception as e:
        print(f"程序执行失败：{e}")