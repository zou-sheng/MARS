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

# 定义目标函数的权重矩阵（这里简单假设为全1）
weights = [[1 for _ in range(cols)] for _ in range(rows)]

# 定义目标函数：矩阵元素的加权和
objective = matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7]
prob += objective

# 行约束：架构约束
# 存储容量
# bypass中keep的张量表示存储在该级buffer，不能超过容量
# register, 存储容量1, 保存weight，weight涉及R, S, C, K, H
prob += matrix[0][0] + matrix[0][1] + matrix[0][4] + matrix[0][5] + matrix[0][7] == 0 

# AccumulationBuffer, 存储容量3072, 保存Output, Output涉及P, Q, K, H, N并且需要乘上前面所有级的值, spatial跟temporal一样处理
prob += matrix[0][2] + matrix[0][3] + matrix[0][5] + matrix[0][6] + matrix[0][7] + \
matrix[1][2] + matrix[1][3] + matrix[1][5] + matrix[1][6] + matrix[1][7] + \
matrix[1][2] + matrix[1][3] + matrix[1][5] + matrix[1][6] + matrix[1][7] <= np.log2(3072)

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

# 求解问题
prob.solve()

# 输出结果
print("Status:", prob.status)
print("Optimal value:", prob.objective.value())
for i in range(rows):
    for j in range(cols):
        print(f"x_{i}_{j} =", matrix[i][j].value())
    
for i in range(rows):
    row_values = []
    for j in range(cols):
        # 对矩阵元素取以 2 为底的指数
        exp_value = 2 ** matrix[i][j].value()
        row_values.append(f"{exp_value:.2f}")  # 保留两位小数
    # 按行输出
    print(" ".join(row_values))