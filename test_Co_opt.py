# 安装依赖库
# pip install pulp

from pulp import LpMinimize, LpProblem, LpVariable, lpSum

def main():
    # 创建问题实例
    prob = LpProblem("Minimize_Max_Example", LpMinimize)
    
    # 定义变量 (变量下界默认为0)
    x1 = LpVariable("x1")
    x2 = LpVariable("x2")
    x3 = LpVariable("x3")
    
    # 定义辅助变量t，表示x1, x2, x3中的最大值
    t = LpVariable("t")
    
    # 添加约束条件: t >= x_i 对所有i
    prob += t >= x1
    prob += t >= x2
    prob += t >= x3
    
    # 添加额外约束条件
    prob += x1 + x2 + x3 >= 10  # 变量和至少为10
    prob += x1 <= 5            # x1上限为5
    prob += x2 <= 6            # x2上限为6
    prob += x3 <= 7            # x3上限为7
    
    # 设置目标函数: 最小化t
    prob += t
    print(prob)
    # 求解问题
    prob.solve()
    
    # 输出结果
    print("优化状态:", prob.status)
    print("目标值(最小化的最大值):", prob.objective.value())
    print("最优解:")
    for v in prob.variables():
        print(f"  {v.name} = {v.value()}")
    
    # 验证约束条件
    print("\n约束条件验证:")
    print(f"x1 + x2 + x3 = {x1.value() + x2.value() + x3.value()} >= 10: {x1.value() + x2.value() + x3.value() >= 10}")
    print(f"x1 = {x1.value()} <= 5: {x1.value() <= 5}")
    print(f"x2 = {x2.value()} <= 6: {x2.value() <= 6}")
    print(f"x3 = {x3.value()} <= 7: {x3.value() <= 7}")
    print(f"t = {t.value()} >= x1: {t.value() >= x1.value()}")
    print(f"t = {t.value()} >= x2: {t.value() >= x2.value()}")
    print(f"t = {t.value()} >= x3: {t.value() >= x3.value()}")

if __name__ == "__main__":
    main()
