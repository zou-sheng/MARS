import random
import numpy as np

# --------------------------
# 1. 定义参数候选列表（核心修改：区间→候选值）
# --------------------------
param_candidates = {
    'x1': [500, 800, 1000, 1500, 2000],  # x1 候选值（整数）
    'x2': [8, 16, 24, 32],               # x2 候选值（整数）
    'x3': [8, 16, 24, 32],               # x3 候选值（整数）
    'x4': [0.1, 0.5, 1.0, 1.5, 2.0]      # x4 候选值（浮点数）
}
# 参数顺序（用于输出统一格式）
param_order = ['x1', 'x2', 'x3', 'x4']

# --------------------------
# 2. 优化目标函数（不变，可自定义）
# --------------------------
def loss_function(params):
    x1, x2, x3, x4 = params
    target = 10000  # 目标值（可替换）
    output = x1 * x2 * x3 * x4
    return abs(output - target)  # 损失越小，参数越优

# --------------------------
# 3. 随机搜索核心逻辑（适配候选列表）
# --------------------------
def random_search_candidates(
    param_candidates,    # 参数候选列表
    param_order,         # 参数顺序
    loss_func,           # 损失函数
    n_samples=1000,      # 采样次数（可调整）
    random_seed=42       # 随机种子（可复现）
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    best_loss = float('inf')
    best_params = None
    all_results = []  # 存储所有采样结果（可选）
    
    for i in range(n_samples):
        # 从每个参数的候选列表中随机选择一个值
        sample_params = [
            random.choice(param_candidates[param])
            for param in param_order
        ]
        
        # 计算损失
        current_loss = loss_func(sample_params)
        all_results.append((sample_params, current_loss))
        
        # 更新最优结果
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = sample_params
            # 每100次采样输出进度
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1:4d} | 最优损失：{best_loss:.2f} | 最优参数：{best_params}")
    
    return best_params, best_loss, all_results

# --------------------------
# 4. 执行随机搜索
# --------------------------
if __name__ == "__main__":
    print("=== 候选列表版随机搜索开始 ===")
    best_params, best_loss, all_results = random_search_candidates(
        param_candidates=param_candidates,
        param_order=param_order,
        loss_func=loss_function,
        n_samples=1000  # 采样1000次（可调整）
    )
    
    print("\n=== 随机搜索结果 ===")
    print(f"最优参数 [x1, x2, x3, x4]：{best_params}")
    print(f"最优损失：{best_loss:.2f}")
    print(f"对应输出 x1*x2*x3*x4：{np.prod(best_params):.2f}（目标：10000）")