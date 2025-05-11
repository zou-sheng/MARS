import numpy as np
from collections import deque

# 原始数据
data = np.array([
    [1.00, 1.00, 1.64, 36.00, 1.00, 1.00, 1.00, 1.00],
    [5.00, 1.15, 1.00, 1.00, 5.54, 1.00, 1.00, 1.00],
    [1.00, 4.33, 1.00, 1.00, 1.00, 51.96, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.33, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 2.77, 1.00, 1.00],
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

def compute_constraints(data):
    """计算所有约束条件的值"""
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
    n = data[0][7] * data[1][7] * data[2][7] * data[3][7] * data[4][7] * data[5][7] * data[6][7] * data[7][7]
    
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

def check_constraints(data):
    """检查是否满足所有约束条件"""
    constraints = compute_constraints(data)
    
    # 检查容量约束
    if constraints['reg_cap'] > target_reg_cap:
        return False
    if constraints['acc_cap'] > target_acc_cap:
        return False
    if constraints['wei_cap'] > target_wei_cap:
        return False
    if constraints['inp_cap'] > target_inp_cap:
        return False
    if constraints['gb_cap'] > target_gb_cap:
        return False
    if constraints['sp_1'] > target_sp_1:
        return False
    if constraints['sp_2'] > target_sp_2:
        return False
    
    # 检查维度约束
    if constraints['r'] < target_r:
        return False
    if constraints['s'] < target_s:
        return False
    if constraints['p'] < target_p:
        return False
    if constraints['q'] < target_q:
        return False
    if constraints['c'] < target_c:
        return False
    if constraints['k'] < target_k:
        return False
    if constraints['h'] < target_h:
        return False
    if constraints['n'] < target_n:
        return False
    
    return True

def maximize_row_with_constraint(data, row, constraint_func, target, max_iterations=10000):
    """最大化指定行，同时满足特定约束，保持原始值为1的位置不变"""
    # 找出原始值为1的位置
    fixed_positions = np.where(data[row] == 1)[0]
    
    upper_bounds = np.ceil(data).astype(int)
    best_data = data.copy()
    best_objective = np.prod(data[row])
    best_constraints_satisfied = check_constraints(data)
    
    queue = deque([data])
    visited = set()
    visited.add(tuple(data.flatten()))
    
    iterations = 0
    while queue and iterations < max_iterations:
        current_data = queue.popleft()
        current_objective = np.prod(current_data[row])
        
        # 检查是否满足约束
        if constraint_func(current_data) <= target:
            if current_objective > best_objective:
                best_data = current_data.copy()
                best_objective = current_objective
                best_constraints_satisfied = True
                print(f"找到更好的解，目标函数值: {best_objective}")
        
        for j in range(len(current_data[row])):
            # 跳过原始值为1的位置
            if j in fixed_positions:
                continue
                
            new_data = current_data.copy()
            # 从向上取整的值开始，逐渐减一，且保证值至少为1
            for value in range(upper_bounds[row][j], 0, -1):
                new_data[row][j] = value
                if np.all(new_data > 0):
                    data_tuple = tuple(new_data.flatten())
                    if data_tuple not in visited:
                        visited.add(data_tuple)
                        queue.append(new_data)
        
        iterations += 1
    
    print(f"行最大化完成，共执行 {iterations} 次迭代")
    return best_data

def integerize_with_staged_optimization(data):
    """分阶段优化：按顺序修改每一行"""
    # 第一阶段：修改第一行
    def constraint_row1(data):
        return data[1][0] * data[1][1] * data[1][2] * data[1][3] * data[1][4] * data[1][5] * data[1][6] * data[1][7]
    data = maximize_row_with_constraint(data, 1, constraint_row1, 32)
    
    # 第二阶段：修改第五行
    def constraint_row5(data):
        return data[5][0] * data[5][1] * data[5][2] * data[5][3] * data[5][4] * data[5][5] * data[5][6] * data[5][7]
    data = maximize_row_with_constraint(data, 5, constraint_row5, 32)
    
    # 第三阶段：修改第0行
    data = maximize_row_with_constraint(data, 0, lambda x: compute_constraints(x)['reg_cap'], target_reg_cap)
    
    # 第四阶段：修改第2行
    data = maximize_row_with_constraint(data, 2, lambda x: compute_constraints(x)['acc_cap'], target_acc_cap)
    
    # 第五阶段：修改第3行
    data = maximize_row_with_constraint(data, 3, lambda x: compute_constraints(x)['wei_cap'], target_wei_cap)
    
    # 第六阶段：修改第4行
    data = maximize_row_with_constraint(data, 4, lambda x: compute_constraints(x)['inp_cap'], target_inp_cap)
    
    # 第七阶段：修改第6行
    data = maximize_row_with_constraint(data, 6, lambda x: compute_constraints(x)['gb_cap'], target_gb_cap)
    
    return data

# 执行整数化处理
integer_result = integerize_with_staged_optimization(data)

# 验证约束
final_constraints = compute_constraints(integer_result)

print("\n约束验证:")
for name, value in final_constraints.items():
    target = globals()[f'target_{name}']
    if isinstance(value, np.ndarray):
        value = np.sum(value)
    if name.startswith('sp_') or name.startswith('reg_') or name.startswith('acc_') or name.startswith('wei_') or name.startswith('inp_') or name.startswith('gb_'):
        print(f"{name}: {value:.2f} (目标: ≤{target})")
    else:
        print(f"{name}: {value:.2f} (目标: ≥{target})")

# 计算目标函数值
obj_value = np.prod(integer_result[1]) + np.prod(integer_result[5])
print(f"\n目标函数值: {obj_value}")

# 打印结果
print("\n整数化结果:")
for row in integer_result:
    print(' '.join([f"{x:2f}" for x in row]))
    
# 检查哪些位置的值被修改了
print("\n修改情况:")
modified = False
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] != integer_result[i, j]:
            print(f"位置 [{i}, {j}]: 原值 {data[i, j]:.2f} -> 新值 {integer_result[i, j]}")
            modified = True

if not modified:
    print("所有值均未被修改")
