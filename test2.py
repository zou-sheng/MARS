class A():
    def __init__(self, a):
        self.a = a

    def exchange(self, b):
        tmp = b.a   
        b.a = self.a
        self.a = tmp


a1 = A(5)
a2 = A(10)
a1.exchange(a2)

print(a1.a, a2.a)

all_configs = [
        [[1, 3, 5, 7], [2, 2, 6, 6]],  # 内部处理后：[2,3,6,7]
        [[2, 4, 4, 8], [3, 1, 5, 5]],  # 内部处理后：[3,4,5,8]
        [[0, 5, 3, 9], [1, 3, 7, 4]]   # 内部处理后：[1,5,7,9]
    ]

max_first = [max(col) for col in zip(*[cfg[0] for cfg in all_configs])]
max_second = [max(col) for col in zip(*[cfg[1] for cfg in all_configs])]

print(max_first, max_second)


import numpy as np

# 原始数据维度（例如：[宽度, 高度, 时间长度]）
dimension = [120, 180, 30]  # 假设是一个时空数据，宽120，高180，时间30帧

# 空间分块信息（每行是一种空间分块方式，列对应宽度、高度）
# 例如：2种空间分块方式，分别为 [2,3] 和 [3,2]
spatial_tiles = np.array([
    [2, 3, 2],   # 第1种空间分块：宽度分2块，高度分3块
    [3, 2, 1]    # 第2种空间分块：宽度分3块，高度分2块
])

# 时间分块信息（每行是一种时间分块方式，列对应时间维度）
# 例如：3种时间分块方式，最后一行可能是特殊分块，计算时排除
temporal_tiles = np.array([
    [2, 2, 2],  # 第1种时间分块：时间分2块
    [3, 1, 1],  # 第2种时间分块：时间分3块
    [5, 3, 3]   # 第3种时间分块（最后一行，计算时排除）
])

def compute_last_row(dimension, spatial_tiles, temporal_tiles):

    spatial_prod = np.prod(spatial_tiles, axis=0, dtype=np.float64)
    print(spatial_prod)
    temporal_prod = np.prod(temporal_tiles[:-1], axis=0, dtype=np.float64)
    print(temporal_prod)
    last_row = np.ceil(np.array(dimension) / (spatial_prod*temporal_prod)).astype(int)

    return last_row

# 调用函数
result = compute_last_row(dimension, spatial_tiles, temporal_tiles)
result = np.vstack((temporal_tiles[:-1], result))
print("最后一行的维度：", result)
