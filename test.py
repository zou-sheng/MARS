import numpy as np
from scipy.optimize import minimize

# 原始数据
data = np.array([
    [1.00, 1.00, 1.64, 36.00, 1.00, 1.00, 1.00, 1.00],
    [5.00, 1.15, 1.00, 1.00, 5.54, 1.00, 1.00, 1.00],
    [1.00, 4.33, 1.00, 1.00, 1.00, 51.96, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.33, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 11.55, 2.77, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 21.92, 1.00, 1.00, 1.00, 1.00, 1.00]
])

# 目标约束值
target_reg_cap = 1
target_acc_cap = 3072
target_wei_cap = 32768
target_inp_cap = 8192
target_gb_cap = 65536
target_sp_1 = 32
target_sp_2 = 32
target_r = 5
target_s = 5
target_p = 36
target_q = 36
target_c = 64
target_k = 192
target_h = 1
target_n = 1

def compute_constraints(x):
    """计算所有约束条件的值"""
    data = x.reshape(8, 8)
    
    # 容量约束
    reg_cap = data[0][0] * data[0][1] * data[0][4] * data[0][5] * data[0][6]
    acc_cap = data[0][2] * data[0][3] * data[0][5] * data[0][6] * data[0][7] * \
              data[1][2] * data[1][3] * data[1][5] * data[1][6] * data[1][7] * \
              data[2][2] * data[2][3] * data[2][5] * data[2][6] * data[2][7]
    wei_cap = data[0][0] * data[0][1] * data[0][4] * data[0][5] * data[0][6] * \
              data[1][0] * data[1][1] * data[1][4] * data[1][5] * data[1][6] * \
              data[2][0] * data[2][1] * data[2][4] * data[2][5] * data[2][6] * \
              data[3][0] * data[3][1] * data[3][4] * data[3][5] * data[3][6]
    inp_cap = data[0][0] * data[0][1] * data[0][2] * data[0][3] * data[0][4] * data[0][6] * data[0][7] * \
              data[1][0] * data[1][1] * data[1][2] * data[1][3] * data[1][4] * data[1][6] * data[1][7] * \
              data[2][0] * data[2][1] * data[2][2] * data[2][3] * data[2][4] * data[2][6] * data[2][7] * \
              data[3][0] * data[3][1] * data[3][2] * data[3][3] * data[3][4] * data[3][6] * data[3][7] * \
              data[4][0] * data[4][1] * data[4][2] * data[4][3] * data[4][4] * data[4][6] * data[4][7]
    gb_cap = data[0][2] * data[0][3] * data[0][5] * data[0][6] * data[0][7] * \
             data[1][2] * data[1][3] * data[1][5] * data[1][6] * data[1][7] * \
             data[2][2] * data[2][3] * data[2][5] * data[2][6] * data[2][7] * \
             data[3][2] * data[3][3] * data[3][5] * data[3][6] * data[3][7] * \
             data[4][2] * data[4][3] * data[4][5] * data[4][6] * data[4][7] * \
             data[5][2] * data[5][3] * data[5][5] * data[5][6] * data[5][7] * \
             data[6][2] * data[6][3] * data[6][5] * data[6][6] * data[6][7] + \
             data[0][0] * data[0][1] * data[0][2] * data[0][3] * data[0][4] * data[0][6] * data[0][7] * \
             data[1][0] * data[1][1] * data[1][2] * data[1][3] * data[1][4] * data[1][6] * data[1][7] * \
             data[2][0] * data[2][1] * data[2][2] * data[2][3] * data[2][4] * data[2][6] * data[2][7] * \
             data[3][0] * data[3][1] * data[3][2] * data[3][3] * data[3][4] * data[3][6] * data[3][7] * \
             data[4][0] * data[4][1] * data[4][2] * data[4][3] * data[4][4] * data[4][6] * data[4][7] * \
             data[5][0] * data[5][1] * data[5][2] * data[5][3] * data[5][4] * data[5][6] * data[5][7] * \
             data[6][0] * data[6][1] * data[6][2] * data[6][3] * data[6][4] * data[6][6] * data[6][7]
    sp_1 = data[1][0] * data[1][1] * data[1][2] * data[1][3] * data[1][4] * data[1][5] * data[1][6] * data[1][7]
    sp_2 = data[5][0] * data[5][1] * data[5][2] * data[5][3] * data[5][4] * data[5][5] * data[5][6] * data[5][7]
    
    # 维度约束
    r = data[0][0] * data[1][0] * data[2][0] * data[3][0] * data[4][0] * data[5][0] * data[6][0] * data[7][0]
    s = data[0][1] * data[1][1] * data[2][1] * data[3][1] * data[4][1] * data[5][1] * data[6][1] * data[7][1]
    p = data[0][2] * data[1][2] * data[2][2] * data[3][2] * data[4][2] * data[5][2] * data[6][2] * data[7][2]
    q = data[0][3] * data[1][3] * data[2][3] * data[3][3] * data[4][3] * data[5][3] * data[6][3] * data[7][3]
    c = data[0][4] * data[1][4] * data[2][4] * data[3][4] * data[4][4] * data[5][4] * data[6][4] * data[7][4]
    k = data[0][5] * data[1][5] * data[2][5] * data[3][5] * data[4][5] * data[5][5] * data[6][5] * data[7][5]
    h = data[0][6] * data[1][6] * data[2][6] * data[3][6] * data[4][6] * data[5][6] * data[6][6] * data[7][6]
    n = data[0][7] * data[1][7] * data[2][7] * data[3][7] * data[4][7] * data[5][7] * data[6][7]
    
    return {
        'reg_cap': reg_cap,
        'acc_cap': acc_cap,
        'wei_cap': wei_cap,
        'inp_cap': inp_cap,
        'gb_cap': gb_cap,
        'sp_1': sp_1,
        'sp_2': sp_2,
        'r': r,
        's': s,
        'p': p,
        'q': q,
        'c': c,
        'k': k,
        'h': h,
        'n': n
    }

def objective_function(x):
    """新的目标函数"""
    data = x.reshape(8, 8)
    obj = data[1][0] * data[1][1] * data[1][2] * data[1][3] * data[1][4] * data[1][5] * data[1][6] * data[1][7] * data[5][0] * data[5][1] * data[5][2] * data[5][3] * data[5][4] * data[5][5] * data[5][6] * data[5][7]
    return -1 * obj

def constraint_function(x):
    """约束函数：计算约束违反程度"""
    constraints = compute_constraints(x)
    
    # 容量约束违反（如果超过目标值）
    cap_violation = 0
    cap_violation += max(0, constraints['reg_cap'] - target_reg_cap)
    cap_violation += max(0, constraints['acc_cap'] - target_acc_cap)
    cap_violation += max(0, constraints['wei_cap'] - target_wei_cap)
    cap_violation += max(0, constraints['inp_cap'] - target_inp_cap)
    cap_violation += max(0, constraints['gb_cap'] - target_gb_cap)
    cap_violation += max(0, constraints['sp_1'] - target_sp_1)
    cap_violation += max(0, constraints['sp_2'] - target_sp_2)
    
    # 维度约束违反（如果小于目标值）
    dim_violation = 0
    dim_violation += max(0, target_r - constraints['r'])
    dim_violation += max(0, target_s - constraints['s'])
    dim_violation += max(0, target_p - constraints['p'])
    dim_violation += max(0, target_q - constraints['q'])
    dim_violation += max(0, target_c - constraints['c'])
    dim_violation += max(0, target_k - constraints['k'])
    dim_violation += max(0, target_h - constraints['h'])
    dim_violation += max(0, target_n - constraints['n'])
    
    return cap_violation + dim_violation

def integerize_with_constraints(data):
    """使用优化方法将数据整数化并满足约束条件"""
    # 初始猜测值（原始数据）
    initial_guess = data.flatten()
    
    # 定义约束：约束违反必须为0
    constraints = ({'type': 'eq', 'fun': lambda x: constraint_function(x)})
    
    # 定义边界：每个值必须为整数
    bounds = [(max(1, int(val) - 10), int(val) + 10) for val in initial_guess]
    
    # 使用优化方法求解
    result = minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 10000, 'disp': True}
    )
    
    # 舍入为整数
    integer_result = np.round(result.x).astype(int).reshape(8, 8)
    
    # 验证约束
    final_constraints = compute_constraints(integer_result)
    
    print("\n约束验证:")
    for name, value in final_constraints.items():
        target = globals()[f'target_{name}']
        if name.startswith('sp_') or name.startswith('reg_') or name.startswith('acc_') or name.startswith('wei_') or name.startswith('inp_') or name.startswith('gb_'):
            print(f"{name}: {value:.2f} (目标: ≤{target})")
        else:
            print(f"{name}: {value:.2f} (目标: ≥{target})")
    
    return integer_result

# 执行整数化处理
integer_result = integerize_with_constraints(data)

# 打印结果
print("\n整数化结果:")
for row in integer_result:
    print(' '.join([f"{x:4d}" for x in row]))
