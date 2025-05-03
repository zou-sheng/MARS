from pulp import LpMaximize, LpProblem, LpVariable
import math

rows = 8 # 存储层次+并行层次
cols = 8 # 问题维度R, S, P, Q, C, K, H, N

# 创建一个最大化问题
prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

# 创建矩阵变量，每个元素是一个非负的连续变量
matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

# 最内层优先级最高，spatial比temporal高, Eyeriss的spatial层次在3和5
total_compute = 8 * (matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][5] + matrix[3][6] + matrix[3][7]) + \
7 * (matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7]) + \
6 * (matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][5] + matrix[0][6] + matrix[0][7]) + \
5 * (matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6] + matrix[1][7]) + \
4 * (matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][5] + matrix[2][6] + matrix[2][7]) + \
3 * (matrix[4][0] + matrix[4][1] + matrix[4][2] + matrix[4][3] + matrix[4][4] + matrix[4][5] + matrix[4][6] + matrix[4][7]) + \
2 * (matrix[6][0] + matrix[6][1] + matrix[6][2] + matrix[6][3] + matrix[6][4] + matrix[6][5] + matrix[6][6] + matrix[6][7]) + \
1 * (matrix[7][0] + matrix[7][1] + matrix[7][2] + matrix[7][3] + matrix[7][4] + matrix[7][5] + matrix[7][6] + matrix[7][7])


prob += total_compute

# 行约束：架构约束
# 存储容量
# bypass中keep的张量表示存储在该级buffer，不能超过容量
# PsumRegFile, 存储容量16, 保存output，output涉及P, Q, K, H, N
prob += matrix[0][2] + matrix[0][3] + matrix[0][5] + matrix[0][6] + matrix[0][7] == math.log2(16)

# WeightRegFile, 存储容量192, wight, wight涉及R, S, C, K, H并且需要乘上前面所有级的值
prob += matrix[0][0] + matrix[0][1] + matrix[0][4] + matrix[0][5] + matrix[0][6] + \
matrix[1][0] + matrix[1][1] + matrix[1][4] + matrix[1][5] + matrix[1][6] <= math.log2(192)

# InputRegFile, 存储容量12，保存input, input涉及R, S, P, Q, C, H, N并且需要乘上前面所有级的值
prob += matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][6] + matrix[0][7] + \
matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][6] + matrix[1][7] + \
matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][6] + matrix[2][7] <= math.log2(12)


# DummyBuffer, 存储容量4，都不保存, spatial跟temporal一样处理
# 如何处理？

# GlobalBuffer, 存储容量131072，保存input和output, spatial跟temporal一样处理
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
matrix[6][0] + matrix[6][1] + matrix[6][2] + matrix[6][3] + matrix[6][4] + matrix[6][6] + matrix[6][7] <= 2*math.log2(131072/2)

# 并行容量
# 并行量直接将各级spatial相乘，由于同一级存储中spatial在temporal前
prob += matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][5] + matrix[3][6] + matrix[3][7] + \
matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= math.log2(256)

# DummyBuffer并行容量16
prob += matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][5] + matrix[3][6] + matrix[3][7] <= math.log2(16)

# GlobalBuffer并行容量16
prob += matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6] + matrix[5][7] <= math.log2(16)

# 列约束：维度大小
col_upper_bounds = [0, 0, math.log2(14), math.log2(14), math.log2(1024), math.log2(512), 0, 0] # 维度值
for j in range(cols):
    col_sum = sum(matrix[i][j] for i in range(rows))
    prob += col_upper_bounds[j] == col_sum

solutions = []

prob.solve()
solution = [[matrix[i][j].value() for j in range(cols)] for i in range(rows)]

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