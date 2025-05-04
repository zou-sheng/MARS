import os
import logging
import yaml
import math
from functools import reduce
from collections import defaultdict, OrderedDict
import subprocess
import random

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)

def run_timeloop(arch, prob, mapp, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['/home/mingchuan/Desktop/zousheng/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-model', str(arch), str(prob), str(mapp)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_timeloop> timeloop-model {} {} {}'.format(arch, prob, mapp))
        return True
    except:
        return False

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def get_factors(n):
    return list(reduce(list.__add__,
                        ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def get_prime_factors(num):
    i = 2
    factors = defaultdict(int)
    while i * i <= num:
        while num % i == 0:
            factors[i] += 1
            num //= i
        i += 1
    if num > 1:
        factors[num] += 1
    return factors

def generate_factors_list(m, n):
    if n < 1:
        print("n 必须大于等于1")
        return []
    
    if m == 0:
        print("m 为0时，所有因数必须为0")
        return [0] * n
    
    if m == 1:
        return [1] * n
    
    factors = get_prime_factors(m)

    result = [1] * n  # 初始化结果列表为 n 个1
    random.shuffle(factors)  # 随机打乱因数
    
    for prime, count in factors.items():
        for _ in range(count):
            index = random.randint(0, n-1)
            result[index] *= prime
    
    return result

def generate_similar_factor_list(pattern, target):
    # 计算模式中所有元素的乘积
    pattern_product = 1
    for num in pattern:
        pattern_product *= num
    # 计算目标值与模式乘积的比值
    ratio = target / pattern_product
    # 找到模式中最大元素的索引
    max_index = pattern.index(max(pattern))
    # 复制模式列表
    result = pattern.copy()
    # 调整最大元素的值
    result[max_index] = int(result[max_index] * ratio)
    # 重新计算调整后列表的乘积
    current_product = 1
    for num in result:
        current_product *= num
    # 如果乘积不等于目标值，尝试微调
    if current_product != target:
        # 尝试逐个调整元素来达到目标值
        for i in range(len(result)):
            if i != max_index:
                new_ratio = target / (current_product / result[i])
                result[i] = int(new_ratio)
                new_product = 1
                for num in result:
                    new_product *= num
                if new_product == target:
                    break
    return result

def run_cosa(output_path='../cosa_tmp', arth_path='../SpatialAccelerators/Cosa/Simba/arch.yaml', map_path='../SpatialAccelerators/Cosa/Simba/mapspace.yaml', prob_path='../SpatialAccelerators/Cosa/Simba/problem.yaml', cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['cosa', '-o', str(output_path), '-ap', str(arth_path), '-mp', str(map_path), '-pp', str(prob_path)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_cosa> cosa -o {} -ap {} -mp {} -pp {}'.format(output_path, arth_path, map_path, prob_path))
        return True
    except:
        return False

def generate_problem_for_cosa(prob_path, dimension_dict):
    dir_path = os.path.dirname(prob_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(prob_path, 'w') as yaml_file:
        yaml.dump(dimension_dict, yaml_file, default_flow_style=False)

def find_remainders(A, T):
    n = len(A) - 1  # n is the number of b's
    # Calculate constant part
   
    constant_part = (A[0] - 1) * np.prod(A[1:])  # 计算常量部分
    while  T - constant_part <= 0:
        A[0] = A[0] - 1
        constant_part = (A[0] - 1) * np.prod(A[1:])

    required_sum = T - constant_part  # 需要的和
    results = []

    # 使用递归穷举 b1, b2, ..., bn
    def backtrack(index, current_b):
        if index == n:  # 已经填充完 b1 到 bn
            # 计算当前和
            current_sum = sum((current_b[i] - 1) * np.prod(A[i + 2:]) for i in range(n-1)) + current_b[n-1]

            if current_sum == required_sum:
                results.append(current_b.copy()[::-1] + [A[0]])  # 存储满足条件的 b
            return 
        
        # 迭代 b_index 的可能值
        for b in range(1, A[index + 1] + 1):  # b的范围是从1到a_{i+1}
            current_b[index] = b
            backtrack(index + 1, current_b)


    # 开始回溯
    backtrack(0, [0] * n)
    return results

def expand_factors(factors, target):

    product = 1
    for factor in factors:
        product *= factor
    if product == target:
        return factors, target
    
    # 复制因数列表，避免修改原始列表
    current_factors = factors.copy()
    current_factors.reverse()

    n = len(current_factors)
    # 找到第一个不为1的因数的索引
    start_index = 0
    # for i, factor in enumerate(current_factors):
    #     if factor != 1:
    #         start_index = i
    #         break

    # 从第一个不为1的因数开始扩展，保证乘积不小于目标值
    product = 1
    for factor in current_factors:
        product *= factor
    while product < target:
        current_factors[start_index] += 1
        product = 1
        for factor in current_factors:
            product *= factor

    # 逐级往后缩减
    index = start_index
    while True:
        while product > target:
            current_factors[index] -= 1
            product = 1
            for factor in current_factors:
                product *= factor
            if product < target:
                current_factors[index] += 1
                product = 1
                for factor in current_factors:
                    product *= factor
                break

        index += 1
        if index == n:
            break

        # 重新扩展当前位置的因数，保证乘积不小于目标值
        while product < target:
            current_factors[index] += 1
            product = 1
            for factor in current_factors:
                product *= factor

    product = 1
    for factor in current_factors:
        product *= factor

    current_factors.reverse()
    return current_factors, product

def generate_mapping_for_lpsolver(solution, dimension, targets, types, bypass_data):
    permutations = ['RSPQCKHN'] * len(targets)
    
    rounded_array = [[math.floor(num) for num in row] for row in solution]
    
    # 按列提取出来
    extracted_columns = [list(col) for col in zip(*rounded_array)]

    factors_list = []
    product_list = []
    for i in range(len(dimension)):
        factors, product = expand_factors(extracted_columns[i], dimension[i])
        factors_list.append(factors)
        product_list.append(product)

    factors_dict = {}
    factors_dict['R'] = factors_list[0]
    factors_dict['S'] = factors_list[1]
    factors_dict['P'] = factors_list[2]
    factors_dict['Q'] = factors_list[3]
    factors_dict['C'] = factors_list[4]
    factors_dict['K'] = factors_list[5]
    factors_dict['H'] = factors_list[6]
    factors_dict['N'] = factors_list[7]
    dimension_dict = {}
    dimension_dict['R'] = product_list[0]
    dimension_dict['S'] = product_list[1]
    dimension_dict['P'] = product_list[2]
    dimension_dict['Q'] = product_list[3]
    dimension_dict['C'] = product_list[4]
    dimension_dict['K'] = product_list[5]
    dimension_dict['H'] = product_list[6]
    dimension_dict['N'] = product_list[7]

    mapping = generate_mapping(factors_dict, permutations, targets, types, bypass_data)
    

    return mapping, dimension_dict

def generate_mapping(factors, permutations, targets, types, bypass_data, remainders={}):
    # 构建结果列表
    mapping = {'mapping': []}
    # 遍历将因数、排列、目标和类型组合成字典
    
    for i in range(len(permutations)):
        # 获取当前的因数组合
        current_permutation = permutations[i]
        current_target = targets[i]
        current_type = types[i]
        
        # 获取因数的值
        factors_values = {}
        for char in current_permutation:
            factors_values[char] = factors[char][i] if len(factors[char]) > 0 else 0
        # 组合成字符串形式
        factors_str_parts = []  # 使用列表来暂存每个部分
        for key in factors_values:
            if key in remainders  and remainders[key] != [] and remainders[key][0][i] != factors_values[key]:
                # 添加格式化字符串到列表
                factors_str_parts.append(f'{key}={factors_values[key]},{remainders[key][0][i]}')
            else:
                factors_str_parts.append(f'{key}={factors_values[key]}')

        # 使用 join 方法将所有部分合并并去除末尾的空格
        factors_str = ' '.join(factors_str_parts)
        # 添加到结果中
        mapping['mapping'].append({
            'factors': factors_str,
            'permutation': current_permutation,
            'target': current_target,
            'type': current_type
        })

    # 添加旁路信息
    mapping['mapping'].extend(bypass_data)

    # 返回结果
    return mapping

def parse_timeloop_output(file_path):
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    in_summary_section = False
    summary = {}
    for line in lines:
        line = line.strip()
        
        # 检测是否进入 Summary Stats 部分
        if line.startswith("Summary Stats"):
            in_summary_section = True
        
        # 若在 Summary Stats 中，查找 Cycles
        if in_summary_section:
            if line.startswith("Cycles:"):
                cycles = float(line.split(":")[1].strip())  # 提取 cycles 值
                summary['cycles'] = cycles
            elif line.startswith("Energy:"):
                energy_str = line.split(':')[1].strip().replace('uJ', '')  # 去掉 uJ
                energy = float(energy_str)  # 转换为浮点数
                summary['energy'] = energy
            elif line.startswith("Utilization:"):
                utilization_str = line.split(':')[1].strip().replace('%', '')  # 去掉百分号
                utilization = float(utilization_str) / 100  # 转换为小数
                summary['utilization'] = utilization
            elif line.startswith("EDP(J*cycle):"):
                EDP = float(line.split(":")[1].strip())  
                summary['EDP'] = EDP
            elif line.startswith("GFLOPs (@1GHz)"):
                GFLOPs = float(line.split(":")[1].strip())  
                summary['GFLOPs'] = GFLOPs
    
    return summary


if __name__ == "__main__":
    print(get_factors(16))
    print(get_prime_factors(16).keys())
    print(generate_factors_list(15, 3))
    print(run_cosa())