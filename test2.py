import numpy as np
import random
from copy import deepcopy

# --------------------------
# 配置参数
# --------------------------
POPULATION_SIZE = 50  # 种群大小
NUM_MATRICES = 50     # 每个个体包含20个矩阵
MATRIX_SHAPE = (3, 3) # 每个矩阵的形状（3x3）
MAX_GENERATIONS = 500  # 最大迭代代数
MUTATION_RATE = 0.1   # 变异概率
ELITISM_RATE = 0.1    # 精英保留比例（保留最优个体的比例）

# --------------------------
# 工具函数：生成随机矩阵
# --------------------------
def generate_random_matrix(shape):
    """生成随机矩阵（元素范围1-10）"""
    return np.random.randint(1, 11, size=shape)

# --------------------------
# 评估函数：单独评估每个矩阵
# --------------------------
def evaluate_matrix(matrix):
    """评估单个矩阵的评分（示例：矩阵元素和的平方根，值越大越好）"""
    return np.sqrt(np.sum(matrix))

def evaluate_individual(individual):
    """评估个体：返回总评分和每个矩阵的单独评分"""
    matrix_scores = [evaluate_matrix(mat) for mat in individual]
    total_score = sum(matrix_scores)  # 总评分用于个体选择
    return total_score, matrix_scores

# --------------------------
# 遗传算法核心逻辑
# --------------------------
def genetic_algorithm():
    # 1. 初始化种群
    population = []
    for _ in range(POPULATION_SIZE):
        # 每个个体是包含20个随机矩阵的列表
        individual = [generate_random_matrix(MATRIX_SHAPE) for _ in range(NUM_MATRICES)]
        population.append(individual)
    
    # 初始化矩阵级最优记录（每个位置的最优矩阵和对应评分）
    best_matrices = [generate_random_matrix(MATRIX_SHAPE) for _ in range(NUM_MATRICES)]
    best_matrix_scores = [-float('inf') for _ in range(NUM_MATRICES)]
    
    # 2. 迭代进化
    for generation in range(MAX_GENERATIONS):
        # 评估种群中所有个体
        evaluated_pop = []
        for ind in population:
            total_score, mat_scores = evaluate_individual(ind)
            evaluated_pop.append((ind, total_score, mat_scores))
        
        # 按总评分排序（降序）
        evaluated_pop.sort(key=lambda x: x[1], reverse=True)
        current_best_total = evaluated_pop[0][1]
        
        # 3. 更新矩阵级最优记录
        for mat_idx in range(NUM_MATRICES):
            # 找出当前代中第mat_idx个矩阵的最高评分
            current_gen_best_score = -float('inf')
            current_gen_best_mat = None
            for (ind, _, mat_scores) in evaluated_pop:
                if mat_scores[mat_idx] > current_gen_best_score:
                    current_gen_best_score = mat_scores[mat_idx]
                    current_gen_best_mat = ind[mat_idx]
            
            # 与历史最优比较，保留更优矩阵
            if current_gen_best_score > best_matrix_scores[mat_idx]:
                best_matrix_scores[mat_idx] = current_gen_best_score
                best_matrices[mat_idx] = deepcopy(current_gen_best_mat)
        
        # 打印当前代信息
        if generation % 10 == 0:
            print(f"第{generation}代 | 种群最优总评分: {current_best_total:.2f} | "
                  f"矩阵最优评分均值: {np.mean(best_matrix_scores):.2f}")
        
        # 4. 生成下一代
        next_population = []
        
        # 精英保留：直接保留前ELITISM_RATE比例的个体
        elite_count = int(ELITISM_RATE * POPULATION_SIZE)
        for i in range(elite_count):
            next_population.append(deepcopy(evaluated_pop[i][0]))
        
        # 剩余个体通过交叉和变异生成
        while len(next_population) < POPULATION_SIZE:
            # 选择父母（轮盘赌选择，基于总评分）
            total_scores = [x[1] for x in evaluated_pop]
            sum_scores = sum(total_scores)
            probs = [s / sum_scores for s in total_scores]  # 选择概率与评分正相关
            parent1 = random.choices(evaluated_pop, weights=probs)[0][0]
            parent2 = random.choices(evaluated_pop, weights=probs)[0][0]
            
            # 交叉：结合父母基因，并有概率引入矩阵级最优
            child = []
            for mat_idx in range(NUM_MATRICES):
                # 30%概率直接继承该位置的历史最优矩阵
                if random.random() < 0.3:
                    child.append(deepcopy(best_matrices[mat_idx]))
                else:
                    # 否则随机选择父母中的一个矩阵
                    child.append(deepcopy(random.choice([parent1[mat_idx], parent2[mat_idx]])))
            
            # 变异：随机微调矩阵元素
            for mat_idx in range(NUM_MATRICES):
                if random.random() < MUTATION_RATE:
                    # 随机选择一个位置，增减1（保持元素在1-10范围内）
                    row = random.randint(0, MATRIX_SHAPE[0]-1)
                    col = random.randint(0, MATRIX_SHAPE[1]-1)
                    child[mat_idx][row, col] = np.clip(
                        child[mat_idx][row, col] + random.choice([-1, 1]),
                        1, 10
                    )
            
            next_population.append(child)
        
        # 更新种群
        population = next_population
    
    # 进化结束，返回矩阵级最优结果
    return best_matrices, best_matrix_scores

# --------------------------
# 运行并测试
# --------------------------
if __name__ == "__main__":
    # 运行遗传算法
    final_best_matrices, final_best_scores = genetic_algorithm()
    
    # 打印结果
    print("\n最终每个矩阵的最优评分：")
    for i in range(NUM_MATRICES):
        print(f"矩阵{i+1} | 评分: {final_best_scores[i]:.2f} | 矩阵:\n{final_best_matrices[i]}\n")
    print(f"所有矩阵最优评分总和：{sum(final_best_scores):.2f}")
