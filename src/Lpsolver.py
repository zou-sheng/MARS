from pulp import LpMaximize, LpProblem, LpVariable
import numpy as np

# 为了解决变量之间存在乘法的问题，所有变量都是取log2之后的值
# 矩阵的行数和列数
rows = 8 # 存储层次+并行层次
cols = 8 # 问题维度R, S, P, Q, C, K, H, N

# 创建一个最大化问题
prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

# 创建矩阵变量，每个元素是一个非负的连续变量
matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]



# 定义目标函数：矩阵元素的加权和
# objective = matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
# matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7]
# prob += objective

# cosa中cycle计算分三个层次，最底层到global层次, global层次，global层次到dram层次

# 最内层优先级最高，spatial比temporal高
total_compute = 8 * (matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6] + matrix[1][7]) + \
7 * (matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7]) + \
6 * (matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][5] + matrix[0][6] + matrix[0][7]) + \
5 * (matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][5] + matrix[2][6] + matrix[2][7]) + \
4 * (matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][5] + matrix[3][6] + matrix[3][7]) + \
3 * (matrix[4][0] + matrix[4][1] + matrix[4][2] + matrix[4][3] + matrix[4][4] + matrix[4][5] + matrix[4][6] + matrix[4][7]) + \
2 * (matrix[6][0] + matrix[6][1] + matrix[6][2] + matrix[6][3] + matrix[6][4] + matrix[6][5] + matrix[6][6] + matrix[6][7]) + \
1 * (matrix[7][0] + matrix[7][1] + matrix[7][2] + matrix[7][3] + matrix[7][4] + matrix[7][5] + matrix[7][6] + matrix[7][7])

prob += total_compute

# 行约束：架构约束
# 存储容量
# bypass中keep的张量表示存储在该级buffer，不能超过容量
# register, 存储容量1, 保存weight，weight涉及R, S, C, K, H
prob += matrix[0][0] + matrix[0][1] + matrix[0][4] + matrix[0][5] + matrix[0][7] == 0 

# AccumulationBuffer, 存储容量3072, 保存Output, Output涉及P, Q, K, H, N并且需要乘上前面所有级的值, spatial跟temporal一样处理
prob += matrix[0][2] + matrix[0][3] + matrix[0][5] + matrix[0][6] + matrix[0][7] + \
matrix[1][2] + matrix[1][3] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
matrix[2][2] + matrix[2][3] + matrix[2][5] + matrix[2][6] + matrix[2][7] <= np.log2(3072)

# WeightBuffer, 存储容量32768，保存weight, weight涉及R, S, C, K, H并且需要乘上前面所有级的值, spatial跟temporal一样处理
prob += matrix[0][0] + matrix[0][1] + matrix[0][4] + matrix[0][5] + matrix[0][7] + \
matrix[1][0] + matrix[1][1] + matrix[1][4] + matrix[1][5] + matrix[1][7] + \
matrix[2][0] + matrix[2][1] + matrix[2][4] + matrix[2][5] + matrix[2][7] + \
matrix[3][0] + matrix[3][1] + matrix[3][4] + matrix[3][5] + matrix[3][7] <= np.log2(32768)

# InputBuffer, 存储容量8192，保存Input, Input涉及R, S, P, Q, C, K, H, N并且需要乘上前面所有级的值, spatial跟temporal一样处理
prob += matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][6] + matrix[0][7] + \
matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][6] + matrix[1][7] + \
matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][6] + matrix[2][7] + \
matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][6] + matrix[3][7] + \
matrix[4][0] + matrix[4][1] + matrix[4][2] + matrix[4][3] + matrix[4][4] + matrix[4][6] + matrix[4][7] <= np.log2(8192)

# GlobalBuffer, 存储容量65536，保存Input和Output, 需要乘上前面所有级的Input和Output涉及的值再相加, spatial跟temporal一样处理
# 加法不好处理，转化为泰勒展开，近似处理
prob += matrix[0][2] + matrix[0][3] + matrix[0][5] + matrix[0][6] + matrix[0][7] + \
matrix[1][2] + matrix[1][3] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
matrix[2][2] + matrix[2][3] + matrix[2][5] + matrix[2][6] + matrix[2][7] + \
matrix[3][2] + matrix[3][3] + matrix[3][5] + matrix[3][6] + matrix[3][7] + \
matrix[4][2] + matrix[4][3] + matrix[4][5] + matrix[4][6] + matrix[4][7] + \
matrix[5][2] + matrix[5][3] + matrix[5][5] + matrix[5][6] + matrix[5][7] + \
matrix[6][2] + matrix[6][3] + matrix[6][5] + matrix[6][6] + matrix[6][7] + \
matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][6] + matrix[0][7] + \
matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][6] + matrix[1][7] + \
matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][6] + matrix[2][7] + \
matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][6] + matrix[3][7] + \
matrix[4][0] + matrix[4][1] + matrix[4][2] + matrix[4][3] + matrix[4][4] + matrix[4][6] + matrix[4][7] + \
matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][6] + matrix[5][7] + \
matrix[6][0] + matrix[6][1] + matrix[6][2] + matrix[6][3] + matrix[6][4] + matrix[6][6] + matrix[6][7] <= 2*np.log2(65536/2)   # 这个等式不严谨，放松了，但没想到更好的办法

# 并行容量
# 并行量直接将各级spatial相乘，由于同一级存储中spatial在temporal前
prob += matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= np.log2(1024)

# 并行度不能超过前面每一级的instance数
# 不能超过register
prob += matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= np.log2(1024)
# 不能超过AccumulationBuffer
prob += matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= np.log2(16)
# 不能超过WeightBuffer
prob += matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= np.log2(16)
# InputBuffer
prob += matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= np.log2(16)



# 列约束：维度大小
col_upper_bounds = [0, 0, np.log2(14), np.log2(14), np.log2(1024), np.log2(512), 0, 0] # 维度值
for j in range(cols):
    col_sum = sum(matrix[i][j] for i in range(rows))
    prob += col_upper_bounds[j] == col_sum

# 求解问题并找到k个解
k = 3  # 你可以修改为你想要的解的数量
solutions = []
for _ in range(k):
    prob.solve()
    if prob.status == 1:
        solution = [[matrix[i][j].value() for j in range(cols)] for i in range(rows)]
        solutions.append(solution)
        # 添加约束以排除当前解
        new_constraint = None
        for i in range(rows):
            for j in range(cols):
                current_constraint = matrix[i][j] != solution[i][j]
                if new_constraint is None:
                    new_constraint = current_constraint
                else:
                    new_constraint = new_constraint | current_constraint
        prob += new_constraint
    else:
        break

# 输出结果
for idx, solution in enumerate(solutions):
    print(f"Solution {idx + 1}:")
    print("Status:", prob.status)
    print("Optimal value:", prob.objective.value())
    for i in range(rows):
        for j in range(cols):
            print(f"x_{i}_{j} =", solution[i][j])
    for i in range(rows):
        row_values = []
        for j in range(cols):
            # 对矩阵元素取以 2 为底的指数
            exp_value = 2 ** solution[i][j]
            row_values.append(f"{exp_value:.2f}")  # 保留两位小数
        # 按行输出
        print(" ".join(row_values))


# 实现一个从解到合法映射的算法，核心是确定如何让比规模小的因数列表扩展到该规模。
# 例如[1,5,181]和1024, 先从最外层扩展，保证乘积刚好比规模大，然后逐级往后缩减，保证每次都是正好大于规模，直到最后一层修改后，正好等于该规模

def expand_factors(factors, target):
    # 复制因数列表，避免修改原始列表
    current_factors = factors.copy()
    n = len(current_factors)
    # 最外层扩展，保证乘积刚好比规模大
    product = 1
    for factor in current_factors:
        product *= factor
    while product <= target:
        current_factors[0] += 1
        product = 1
        for factor in current_factors:
            product *= factor

    # 逐级往后缩减
    index = 0
    while True:
        while product > target:
            current_factors[index] -= 1
            product = 1
            for factor in current_factors:
                product *= factor
            if product <= target:
                current_factors[index] += 1
                product = 1
                for factor in current_factors:
                    product *= factor
                break

        index += 1
        if index == n:
            break

        # 重新扩展当前位置的因数，保证乘积刚好比规模大
        while product <= target:
            current_factors[index] += 1
            product = 1
            for factor in current_factors:
                product *= factor

    return current_factors


factors = [1, 5, 1, 181]
target = 1024
result = expand_factors(factors, target)
print("扩展后的因数列表:", result)

# 证明
# 设给定的因数列表为 \(F = [f_1, f_2, \cdots, f_n]\)，目标规模为 T。当前因数列表为 \(C = [c_1, c_2, \cdots, c_n]\)，初始时 \(C = F\)。
# 计算当前因数列表的乘积 \(P=\prod_{i = 1}^{n}c_i\)。当 \(P\leq T\) 时，对最外层因数 \(c_1\) 进行更新，即 \(c_1=c_1 + 1\)，然后重新计算 \(P=\prod_{i = 1}^{n}c_i\)，直到 \(P>T\)。
# 对于第 j 步（j 从 1 到 n），当 \(j = 1\) 时已经完成初始扩展。从 \(j = 2\) 开始：缩减阶段：当 \(P>T\) 时，更新 \(c_j=c_j - 1\)，并重新计算 \(P=\prod_{i = 1}^{n}c_i\)，直到 \(P\leq T\)，然后再将 \(c_j=c_j + 1\) 以保证 \(P>T\)。
# 后续扩展阶段：如果 \(j < n\)，继续扩展下一个因数 \(c_{j + 1}\)，当 \(P\leq T\) 时，更新 \(c_{j+1}=c_{j + 1}+1\)，并重新计算 \(P=\prod_{i = 1}^{n}c_i\)，直到 \(P>T\)。
# 这个逐层调整的算法必唯一