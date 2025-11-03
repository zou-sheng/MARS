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