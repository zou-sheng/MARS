from datetime import datetime
from cost_model import Timeloop
import pathlib
from input_objs import Arch, Prob, Mapspace, Mapping
import utils
import random
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD
import math
import copy
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import uuid
import os
import shutil
import numpy as np
from collections import deque
import time
import cvxpy as cp
import bisect


class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, expanded_scope, expanded_dict, parameter_dimension=2, weight_matrix=None, solver='lp'):
        if optim_obj == 'latency':
            self.fitness_obj = ['cycles']
        elif optim_obj == 'all':
            self.fitness_obj = ['EDP', 'energy', 'cycles']
        else:
            self.fitness_obj = [optim_obj]
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.operator_instance = operator_instance
        
        self.report_dir = report_dir 
        self.expanded_scope = expanded_scope
        self.expanded_dict = expanded_dict
        self.mapper = mapper
        self.weight_matrix = weight_matrix
        self.solver = solver

        arch_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, type)).resolve()
        self.accelerator = Arch(arch_path, accelerator, version)
        mapspace_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, 'mapspace')).resolve()
        self.mapspace = Mapspace(mapspace_path)
        prob_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, 'problem')).resolve()
        self.problem = Prob(prob_path)
        self.buffer_tensor_dict = self.mapspace.buffer_tensor_dict
        self.bypass = self.mapspace.bypass
        self.tensor_dimensions = self.problem.get_tensor_dimensions()

        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels = self.accelerator.get_arch_info()

        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels

        self.buf_energy_cost = self.get_default_buffer_energy_cost()

        self.dimension, self.dimension_dict = self.problem.get_problem_info()
        self.expanded_dimension_dict = self.get_expanded_problem()
        self.factors_candidate = [sorted(utils.get_factors(num), reverse=True) for num in self.dimension]
        print(self.factors_candidate)

       
        # 用于存储temporal层级
        self.temporal_level = {}
        # 用于存储spatial层级
        self.spatial_level = {}
        self.spatial_size_list = {}

        self.obj_level = []

        index = 0

        self.targets = []
        self.type = []

        # 对 buffer_name_list 的键按一定顺序排序
        for key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[key]
            if key in self.buffers_with_spmap:
                self.spatial_level[buffer_name] = index
                self.temporal_level[buffer_name] = index + 1
                self.obj_level.append(index)
                index += 2
                self.spatial_size_list[buffer_name] = self.buffer_spmap_cstr[key]
                self.targets.append(buffer_name)
                self.type.append('spatial')
            else:
                self.temporal_level[buffer_name] = index
                index += 1
            self.targets.append(buffer_name)
            self.type.append('temporal')

        for key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[key]
            self.obj_level.append(self.temporal_level[buffer_name])

        self.para_dim = parameter_dimension
        self.num_rows = len(self.temporal_level) + len(self.spatial_level)
        self.num_cols = 8
        self.candidates = [[list(range(1, d+1)) for d in self.dimension] for _ in range(self.num_rows)]

    def shuffle_factor_order(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            # 随机选择一个列表的键
            random_key = random.choice(list(mapping.factor_dict.keys()))

            # 获取随机选择的列表
            list_to_shuffle = mapping.factor_dict[random_key]

            # 打乱列表
            random.shuffle(list_to_shuffle) 
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping



    def mutate_factor(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            # 随机选择一个列表的键
            random_key1, random_key2 = random.sample(list(mapping.factor_dict.keys()), 2)

            # 获取随机选择的列表
            list_to_modify1 = mapping.factor_dict[random_key1]
            list_to_modify2 = mapping.factor_dict[random_key2]

            # 随机选择两个不同的索引
            index1, index2 = random.sample(range(len(list_to_modify1)), 2)

            # 选择这两个数字
            num1 = list_to_modify1[index1]
            num2 = list_to_modify1[index2]
            num3 = list_to_modify2[index1]
            num4 = list_to_modify2[index2]

            # 计算第一个数字的因子
            factors = [i for i in range(1, num1 + 1) if num1 % i == 0]

            # 随机选择一个因子
            random_factor = random.choice(factors)

            # 将因子乘到第二个数字上
            modified_num2 = num2 * random_factor
            modified_num1 = int(num1 / random_factor)
            list_to_modify1[index1] = modified_num1
            list_to_modify1[index2] = modified_num2
            if int(num4 / random_factor) == 0:
                pass
            else:
                list_to_modify2[index1] = num3 * random_factor
                list_to_modify2[index2] = int(num4 / random_factor)
            
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping



    def mutate_permutation(self, mapping, alpha=0.5):
        if random.random() < alpha:
            candidate_permutation = ['RSPQCKNH',    # 循环顺序是内->外
                                    'PQNRSCKH',
                                    'QPNRSCKH',
                                    'SRCPQKNH',
                                    'RSCPQKNH',
                                    'KQRSPCNH',
                                    'KPRSQCNH',
                                    'KSRPQCNH',
                                    'KRSPQCNH',
                                    'HRSPQCKN',
                                    'HPQNRSCK',
                                    'HQPNRSCK',
                                    'HSRCPQKN',
                                    'HRSCPQKN',
                                    'HKQRSPCN',
                                    'HKPRSQCN',
                                    'HKSRPQCN',
                                    'HKRSPQCN',
                                    ]
            # 随机选择一个当前列表中的索引
            index_to_replace = random.randint(0, len(mapping.permutation_list) - 1)

            # # 从候选列表中随机选择一个新的 permutation
            new_permutation = random.choice(candidate_permutation)
            # s = "RSPQCKNH"
            # s_list = list(s)
            # random.shuffle(s_list)
            # new_permutation = "".join(s_list) 
            # 替换选择的 permutation
            mapping.permutation_list[index_to_replace] = new_permutation

    # 随机选择一个维度的扩展
    def mutate_dimemsion(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            for k in mapping.factor_dict.keys():
                if random.random() < alpha:
                    try:
                        new_dim = random.choice(self.expanded_dimension_dict[k])
                      
                        # 跟mapping中factor的个数有关
                        new_fator = utils.generate_similar_factor_list(mapping.factor_dict[k], new_dim)
                        product = 1
                        for num in new_fator:
                            product *= num
                        if product < self.problem[k]:
                            pass
                        else:
                            mapping.factor_dict[k] = new_fator
                    except:
                        pass   
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping

    def generate_prob_for_cosa(self, operator_instance):
        # 要保留的键
        desired_keys = [
            'C', 'Hdilation', 'Hstride', 'K', 'N', 'P', 'Q', 'R', 'S', 'H', 'Wdilation', 'Wstride'
        ]

        # 构建目标字典
        prob_dict = {
            'problem': {
                key: operator_instance[key] for key in desired_keys if key in operator_instance
            }
        }
        prob_dict['problem']['shape'] = 'cnn-layer'

        return prob_dict

    
    def generate_mapping(self, dimension, p, a=1):
        if self.solver == 'lp':
            sol = self.lpsolver(dimension, p)
        elif self.solver == 'qp':
            sol = self.qpsolver(dimension, p, a)
        mapping_dict, dimension_dict = utils.generate_mapping_for_lpsolver2(sol, self.targets, self.type, self.bypass)
        
        prob = copy.deepcopy(self.problem.problem)       
        for key in dimension_dict.keys():
            prob['problem']['instance'][key] = dimension_dict[key]
        return mapping_dict, prob

    def lpsolver(self, dimension_list, p):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

        # 定义目标函数：矩阵元素的加权和
        objective = 0
        if self.para_dim == 1:
            idx = 0
            p_i = 0
            for spatial_name in self.spatial_level:
                obj_sum = 0
                sp_level = self.spatial_level[spatial_name] 
                for c in range(cols):
                    obj_sum += matrix[sp_level][c]
                obj_sum = obj_sum * p[p_i]
                objective += obj_sum
                idx += 1
                p_i += 1


            idx = 0
            # 除DRAM外每一个存储层次的存储容量约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                obj_sum = 0
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    break
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                for l in range(buffer_level+1):
                    for tensor in tensors_list:
                        for dim in self.tensor_dimensions[tensor]:
                            obj_sum += matrix[l][dim]
                obj_sum = obj_sum * p[p_i]
                objective += obj_sum
                idx += 1
                p_i += 1
            
        elif self.para_dim == 2:
            for r in range(rows):
                for c in range(cols):
                    objective += matrix[r][c] * p[r][c]
        
        prob += objective
        
        # 除DRAM外每一个存储层次的存储容量约束
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = 0
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                break
            buffer_level = self.temporal_level[buffer_name]
            tensors_list = self.buffer_tensor_dict[buffer_name]
            for l in range(buffer_level+1):
                for tensor in tensors_list:
                    for dim in self.tensor_dimensions[tensor]:
                        buffer_capacity += matrix[l][dim]
            # print("buffer_capacity: ", buffer_capacity)
            if len(tensors_list) > 0:
                prob += buffer_capacity <= len(tensors_list)*math.log2(self.buffer_size_list[buffer_key]/len(tensors_list))
        
        # 并行容量约束
        for spatial_name in self.spatial_level:
            spatial_capacity = 0
            sp_level = self.spatial_level[spatial_name] 
            for c in range(cols):
                spatial_capacity += matrix[sp_level][c]
            # print("spatial_capacity: ", spatial_capacity)
            prob += spatial_capacity <= math.log2(self.spatial_size_list[spatial_name])

        # 维度约束
        for c in range(cols):
            col_sum = sum(matrix[r][c] for r in range(rows))
            for r in range(rows):
                prob += matrix[r][c] <= math.log2(dimension_list[c])
            prob += col_sum >= math.log2(dimension_list[c])

        prob.solve(PULP_CBC_CMD(msg=0))
        # print(prob)
        
        # 记录开始时间
        # start_time = time.time()
        solution = [[2**matrix[i][j].value() for j in range(cols)] for i in range(rows)]
        print(solution)
        # print("time1: ", time.time()-start_time)
        solution = self._integerize_with_staged_optimization(solution, p, mode='PFM')
        # solution = self._integerize_optimization(solution, p, mode='PFM')
        print(solution)
        # print("time2: ", time.time()-start_time)
        return solution
    
    def qpsolver(self, dimension_list, p, a=10):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N
        n = rows * cols # n为变量维度
        x = cp.Variable(n)
        P = np.diag([2*a] * n)
        q = p.flatten()
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

        constraints = []

        # 1. 除DRAM外每一个存储层次的存储容量约束
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                break
                
            buffer_level = self.temporal_level[buffer_name]
            tensors_list = self.buffer_tensor_dict[buffer_name]
            
            if len(tensors_list) > 0:
                # 计算右侧值
                rhs = len(tensors_list) * math.log2(self.buffer_size_list[buffer_key] / len(tensors_list))
                
                # 构建左侧系数向量
                coeffs = np.zeros((rows, cols))
                for l in range(buffer_level + 1):
                    for tensor in tensors_list:
                        for dim in self.tensor_dimensions[tensor]:
                            coeffs[l][dim] += 1
                
                # 转为1D向量并添加到约束
                flat_coeffs = coeffs.flatten()
                constraints.append((flat_coeffs, rhs, 'leq'))

        # 2. 并行容量约束
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            rhs = math.log2(self.spatial_size_list[spatial_name])
            
            # 构建左侧系数向量
            coeffs = np.zeros((rows, cols))
            for c in range(cols):
                coeffs[sp_level][c] += 1
            
            # 转为1D向量并添加到约束
            flat_coeffs = coeffs.flatten()
            constraints.append((flat_coeffs, rhs, 'leq'))
        
        # 3. 维度约束 - 上下界
        for c in range(cols):
            max_val = math.log2(dimension_list[c])
            for r in range(rows):
                # 下界约束: 0 <= matrix[r][c]
                coeffs = np.zeros((rows, cols))
                coeffs[r][c] = -1  # -matrix[r][c] <= 0
                flat_coeffs = coeffs.flatten()
                constraints.append((flat_coeffs, 0, 'leq'))
                
                # 上界约束: matrix[r][c] <= max_val
                coeffs = np.zeros((rows, cols))
                coeffs[r][c] = 1
                flat_coeffs = coeffs.flatten()
                constraints.append((flat_coeffs, max_val, 'leq'))
        
        # 4. 维度约束 - 列总和
        for c in range(cols):
            min_sum = math.log2(dimension_list[c])
            
            # 列总和约束: sum(matrix[r][c]) >= min_sum
            coeffs = np.zeros((rows, cols))
            for r in range(rows):
                coeffs[r][c] = -1  # -sum(matrix[r][c]) <= -min_sum
            flat_coeffs = coeffs.flatten()
            constraints.append((flat_coeffs, -min_sum, 'leq'))
        
        # 将约束转换为G, h矩阵形式 (Gx <= h)
        num_vars = rows * cols
        num_constraints = len(constraints)
        
        G = np.zeros((num_constraints, num_vars))
        h = np.zeros(num_constraints)
        
        for i, (coeffs, rhs, sense) in enumerate(constraints):
            G[i] = coeffs
            h[i] = rhs

        constraints = [G @ x <= h]

        problem = cp.Problem(objective, constraints)
        # 记录开始时间
        start_time = time.time()
        # 求解并输出结果
        problem.solve(solver=cp.OSQP, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5)
        print("time1: ", time.time()-start_time)
        result_matrix = x.value.reshape(rows, cols)
        power_matrix = np.power(2, result_matrix)
        
        solution = self._integerize_with_staged_optimization5(power_matrix, p, a)
        print(solution)
        print("time2: ", time.time()-start_time)
        return solution

    def _integerize_with_staged_optimization1(self, solution):
        def calculate_remaining_capacity(data, fixed_rows):
            """
            计算在固定行确定后，各个存储层次的剩余容量
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            
            返回:
            一个字典，包含每个存储层次的总容量、已占容量和剩余容量
            """
            result = {}
                     
            # 处理buffer约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                constraint['keep']
                # 跳过DRAM
                if buffer_name == 'DRAM':
                    continue

                # 获取该buffer约束涉及的所有行
                all_involved_rows = list(range(buffer_level + 1))
                
                # 计算buffer约束的已占容量
                used_capacity = 0
                tensors_list = self.buffer_tensor_dict[buffer_name]
                
                for tensor in tensors_list:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            # 如果行已固定，使用其值；否则假设为1
                            if l in fixed_rows:
                                tensor_capacity *= data[l][dim]
                            else:
                                tensor_capacity *= 1  # 假设值为1
                    
                    used_capacity += tensor_capacity
                
                result[buffer_name] = used_capacity            
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """
            检查在固定行确定后，指定存储层次的剩余容量约束是否满足
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            storage_name: 存储层次名称
            
            返回:
            True如果约束满足，False否则
            """
            capacities = calculate_remaining_capacity(data, fixed_rows)
            
            # 检查指定存储层次的剩余容量是否非负
            for buffer_key in sorted(self.buffer_name_list.keys()):                
                buffer_capacity = self.buffer_size_list[buffer_key]
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > buffer_capacity:
                    return False
            
            return True 

               
        def maximize_row_with_constraint(data, row, constraint_func, target, objective_func, fixed_rows):
            """
            递归DFS版本：最大化指定行，同时满足特定约束，只处理原始值不为1的列
            """
            # 找出原始值为1的位置
            fixed_positions = np.where(data[row] == 1)[0]
            # 确定需要处理的列（原始值不为1的列）
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]

            # 按原始值降序排序（关键修改点）
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)

            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index):
                iterations[0] += 1
                nonlocal best_data, best_objective
                # 检查约束条件
                if constraint_func(current_data) <= target:
                    # 检查所有剩余容量约束
                    if remaining_capacity_constraint(current_data, fixed_rows + [row]):
                        current_objective = objective_func(current_data)
                        if current_objective > best_objective:
                            best_data = current_data.copy()
                            best_objective = current_objective
                            # print(f"找到更好的解，目标函数值: {best_objective}")
                        return True

                if col_index >= len(columns_to_process):
                    return False

                # 获取当前要处理的列索引
                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]

                # 从大到小尝试值，找到第一个满足条件的值后停止
                for value in range(original_value, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value

                    # # 检查是否满足基本条件
                    # if np.all(new_data > 0):
                    #     # 检查约束条件
                    #     if constraint_func(new_data) > target:
                    #         continue
                    #     print("new_data2", new_data)
                    # 递归处理下一列
                    flag = bfs(new_data, col_index + 1)

                    if flag:
                        break

            # 从第一列开始DFS
            bfs(data.copy(), 0)

            # print(f"第{row}行最大化完成，目标函数值: {best_objective}，共执行 {iterations[0]} 次迭代")
            return best_data
        
        # 已固定的行列表
        fixed_rows = []

        data = np.ceil(solution).astype(int)
        for spatial_name in self.spatial_level:
            spatial_capacity = self.spatial_size_list[spatial_name]
            sp_level = self.spatial_level[spatial_name] 
            data = maximize_row_with_constraint(data, sp_level, lambda x: np.prod(x[sp_level]), spatial_capacity, lambda x: np.prod(x[sp_level]), fixed_rows)
            fixed_rows.append(sp_level)

        def create_buffer_constraint(buffer_name):
            """创建用于maximize_row_with_constraint的约束函数"""
            def constraint_func(data):
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                buffer_capacity = 0
                
                for tensor in tensors_list:
                    tmp = 1
                    for l in range(buffer_level+1):
                        for dim in self.tensor_dimensions[tensor]:
                            tmp *= data[l][dim]
                    buffer_capacity += tmp
                return buffer_capacity 

            return constraint_func
        
        
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = self.buffer_size_list[buffer_key]
            buffer_name = self.buffer_name_list[buffer_key]
            buffer_level = self.temporal_level[buffer_name]
            if buffer_name == 'DRAM':
                break
            constraint_func = create_buffer_constraint(buffer_name)  
            data = maximize_row_with_constraint(data, buffer_level, constraint_func, buffer_capacity, constraint_func, fixed_rows)
            fixed_rows.append(buffer_level)

        def compute_last_row(data):
            """计算最后一行的值，每一列等于对应的target除以这一列前几行的乘积向上取整"""
            # 计算前n-1行的乘积
            product = np.prod(data[:len(data)-2], axis=0)
            
            # 获取对应的target值
            targets = np.array(self.dimension)
            
            # 计算最后一行的值：target除以乘积，然后向上取整
            last_row = np.ceil(targets / product).astype(int)
            
            # 确保最后一行的每个元素至少为1
            last_row = np.maximum(last_row, 1)
            
            # 更新数据
            new_data = data.copy()
            new_data[len(data)-1] = last_row
            
            # print("最后一行计算完成")
            return new_data

        solution = compute_last_row(data)
        return solution
    
    def _integerize_with_staged_optimization2(self, solution):
        # 已固定的行列表
        fixed_rows = set()

        data = np.ceil(solution).astype(np.int64)
        
        # 处理空间层
        for spatial_name in self.spatial_level:
            spatial_capacity = self.spatial_size_list[spatial_name]
            sp_level = self.spatial_level[spatial_name]
            data = maximize_row_with_constraint(
                self.buffer_info,
                data,
                sp_level,
                lambda x: np.prod(x[sp_level]),
                spatial_capacity,
                lambda x: np.prod(x[sp_level]),
                fixed_rows
            )
            fixed_rows.add(sp_level)
        
        # 处理缓冲层
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = self.buffer_size_list[buffer_key]
            buffer_name = self.buffer_name_list[buffer_key]
            buffer_level = self.temporal_level[buffer_name]
            
            if buffer_name == 'DRAM':
                break
            
            def constraint_func(data):
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                buffer_capacity = 0
                
                for tensor in tensors_list:
                    tmp = 1
                    for l in range(buffer_level + 1):
                        for dim in self.tensor_dimensions[tensor]:
                            tmp *= data[l, dim]
                    buffer_capacity += tmp
                return buffer_capacity
            
            data = maximize_row_with_constraint(
                self.buffer_info,
                data,
                buffer_level,
                constraint_func,
                buffer_capacity,
                constraint_func,
                fixed_rows
            )
            fixed_rows.add(buffer_level)
        
        # 计算最后一行
        solution = compute_last_row(data, self.dimension)
        return solution

    def _integerize_with_staged_optimization3(self, solution):
        def calculate_remaining_capacity(data, fixed_rows):
            """
            计算在固定行确定后，各个存储层次的剩余容量
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            
            返回:
            一个字典，包含每个存储层次的总容量、已占容量和剩余容量
            """
            result = {}
                    
            # 处理buffer约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                
                # 跳过DRAM
                if buffer_name == 'DRAM':
                    continue

                # 获取该buffer约束涉及的所有行
                all_involved_rows = list(range(buffer_level + 1))
                
                # 计算buffer约束的已占容量
                used_capacity = 0
                tensors_list = self.buffer_tensor_dict[buffer_name]
                
                for tensor in tensors_list:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            # 如果行已固定，使用其值；否则假设为1
                            if l in fixed_rows:
                                tensor_capacity *= data[l][dim]
                            else:
                                tensor_capacity *= 1  # 假设值为1
                    
                    used_capacity += tensor_capacity
                
                result[buffer_name] = used_capacity            
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """
            检查在固定行确定后，指定存储层次的剩余容量约束是否满足
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            
            返回:
            True如果约束满足，False否则
            """
            capacities = calculate_remaining_capacity(data, fixed_rows)
            
            # 检查指定存储层次的剩余容量是否非负
            for buffer_key in sorted(self.buffer_name_list.keys()):                
                buffer_capacity = self.buffer_size_list[buffer_key]
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > buffer_capacity:
                    return False
            
            return True 

        def compute_column_upper_bound(data, row, col, fixed_rows, constraint_func, target):
            """
            计算指定列在满足约束条件下的最大可能值
            """
            # 获取原始值作为初始上界
            original_value = data[row][col]
            
            # 尝试二分查找确定上界
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data = data.copy()
                test_data[row][col] = mid
                
                # 检查约束条件
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值
                else:
                    high = mid - 1  # 尝试更小的值
            
            return best_valid

        def maximize_row_with_constraint(data, row, constraint_func, target, objective_func, fixed_rows):
            """
            递归DFS版本：最大化指定行，同时满足特定约束，只处理原始值不为1的列
            """
            # 找出原始值为1的位置
            fixed_positions = np.where(data[row] == 1)[0]
            # 确定需要处理的列（原始值不为1的列）
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]

            # 按原始值降序排序（关键修改点）
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)

            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index):
                iterations[0] += 1
                nonlocal best_data, best_objective
                # 检查约束条件
                if constraint_func(current_data) <= target:
                    # 检查所有剩余容量约束
                    if remaining_capacity_constraint(current_data, fixed_rows + [row]):
                        current_objective = objective_func(current_data)
                        if current_objective > best_objective:
                            best_data = current_data.copy()
                            best_objective = current_objective
                            # print(f"找到更好的解，目标函数值: {best_objective}")
                        return True

                if col_index >= len(columns_to_process):
                    return False

                # 获取当前要处理的列索引
                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]

                # 计算当前列的上界
                upper_bound = compute_column_upper_bound(current_data, row, actual_col, fixed_rows, constraint_func, target)
                
                # 只尝试有效范围内的值
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value

                    # 递归处理下一列
                    flag = bfs(new_data, col_index + 1)

                    if flag:
                        break

            # 从第一列开始DFS
            bfs(data.copy(), 0)

            # print(f"第{row}行最大化完成，目标函数值: {best_objective}，共执行 {iterations[0]} 次迭代")
            return best_data
        
        # 已固定的行列表
        fixed_rows = []

        data = np.ceil(solution).astype(int)
        for spatial_name in self.spatial_level:
            spatial_capacity = self.spatial_size_list[spatial_name]
            sp_level = self.spatial_level[spatial_name] 
            data = maximize_row_with_constraint(data, sp_level, lambda x: np.prod(x[sp_level]), spatial_capacity, lambda x: np.prod(x[sp_level]), fixed_rows)
            fixed_rows.append(sp_level)

        def create_buffer_constraint(buffer_name):
            """创建用于maximize_row_with_constraint的约束函数"""
            def constraint_func(data):
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                buffer_capacity = 0
                
                for tensor in tensors_list:
                    tmp = 1
                    for l in range(buffer_level+1):
                        for dim in self.tensor_dimensions[tensor]:
                            tmp *= data[l][dim]
                    buffer_capacity += tmp
                return buffer_capacity 

            return constraint_func
        
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = self.buffer_size_list[buffer_key]
            buffer_name = self.buffer_name_list[buffer_key]
            buffer_level = self.temporal_level[buffer_name]
            if buffer_name == 'DRAM':
                break
            constraint_func = create_buffer_constraint(buffer_name)  
            data = maximize_row_with_constraint(data, buffer_level, constraint_func, buffer_capacity, constraint_func, fixed_rows)
            fixed_rows.append(buffer_level)

        def compute_last_row(data):
            """计算最后一行的值，每一列等于对应的target除以这一列前几行的乘积向上取整"""
            # 计算前n-1行的乘积
            product = np.prod(data[:len(data)-2], axis=0)
            
            # 获取对应的target值
            targets = np.array(self.dimension)
            
            # 计算最后一行的值：target除以乘积，然后向上取整
            last_row = np.ceil(targets / product).astype(int)
            
            # 确保最后一行的每个元素至少为1
            last_row = np.maximum(last_row, 1)
            
            # 更新数据
            new_data = data.copy()
            new_data[len(data)-1] = last_row
            
            # print("最后一行计算完成")
            return new_data

        solution = compute_last_row(data)
        return solution

    def _integerize_with_staged_optimization4(self, solution):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def maximize_row_with_constraint(data, row, constraint_func, target, objective_func, fixed_rows):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data, best_objective
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    current_objective = objective_func(current_data)
                    if current_objective > best_objective:
                        best_data = current_data.copy()
                        best_objective = current_objective
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        # 固定行优化流程
        fixed_rows = []
        data = np.ceil(solution).astype(int)
        
        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            data = maximize_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), spatial_capacity, 
                lambda x: np.prod(x[sp_level]), fixed_rows
            )
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
                
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            constraint_func = create_buffer_constraint(buffer_name)
            data = maximize_row_with_constraint(
                data, buffer_level, constraint_func, buffer_capacity, constraint_func, fixed_rows
            )
            fixed_rows.append(buffer_level)
        
        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    def _integerize_with_staged_optimization(self, solution, p, mode='IFM'):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            # low, high = 1, math.floor(original_value * 1.1)
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def maximize_row_with_constraint(data, row, constraint_func, target, fixed_rows, p_row):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data, best_objective
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    current_objective = np.sum(p_row * current_data[row])
                    # current_objective = np.sum(p_row * np.log2(current_data[row]))
                    # current_objective = np.sum(p_row * np.log2(current_data[row]) + a * np.log2(current_data[row])**2)
                    if current_objective > best_objective:
                        best_data = current_data.copy()
                        best_objective = current_objective
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                if mode == 'IFM':
                    values_to_try = range(upper_bound, 0, -1)
                else: # PFM
                    values_to_try = [v for v in self.factors_candidate[actual_col] if v <= upper_bound and v <= original_value]
                
                for value in values_to_try:
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        
        # 固定行优化流程
        fixed_rows = []
        
        if mode == 'IFM':
            data = np.ceil(solution).astype(int)
        else:
            solution = np.asarray(solution)
            rows, cols = solution.shape
            # 初始化data数组
            data = np.empty((rows, cols), dtype=int)
            for col in range(cols):
                col_values = solution[:, col]
                # 按从小到大排序（便于找到第一个大于val的最小值）
                col_candidates = sorted(self.factors_candidate[col])
                for i in range(rows):
                    val = solution[i, col]
                    # # 直接处理 val 为 1.0 的情况
                    # if val == 1.0:
                    #     data[i, col] = 1  # 直接赋值为 1
                    #     continue  # 跳过当前循环，进入下一个
                    val = col_values[i]
                    # 查找第一个大于val的候选值，添加默认值避免异常
                    try:
                        # data[i, col] = next(c for c in col_candidates if c >= val)
                        data[i, col] = next(c for c in col_candidates if c > val)
                    except StopIteration:
                        # 处理没有找到符合条件的值的情况
                        # 可以根据业务需求设置默认值，这里示例用最大值
                        data[i, col] = max(col_candidates) if col_candidates else 1


            # solution = np.asarray(solution)
            # rows, cols = solution.shape
            # # 初始化 data 数组
            # data = np.empty((rows, cols), dtype=int)
            
            # for col in range(cols):
            #     col_candidates = sorted(self.factors_candidate[col])  # 确保候选值是有序的
                
            #     for i in range(rows):
            #         val = solution[i, col]
                    
            #         # 直接处理 val 为 1.0 的情况
            #         if val == 1.0:
            #             data[i, col] = 1  # 直接赋值为 1
            #             continue  # 跳过当前循环，进入下一个
                    
            #        # 使用 bisect_right 查找第一个大于等于 val 的候选值
            #         idx = bisect.bisect_right(col_candidates, val)
            #         if idx < len(col_candidates):
            #             data[i, col] = col_candidates[idx]  # 找到的值是第一个大于 val 的值
            #         else:
            #             # 处理没有找到符合条件的值的情况
            #             data[i, col] = max(col_candidates) if col_candidates else 1
        
        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            p_row = p[sp_level]  # 获取当前行的权重
            data = maximize_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), 
                spatial_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
                
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            p_row = p[buffer_level]  # 获取当前行的权重
            constraint_func = create_buffer_constraint(buffer_name)
            data = maximize_row_with_constraint(
                data, buffer_level, 
                constraint_func, 
                buffer_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(buffer_level)
        
        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    # 失败的代码不考虑用 
    def _integerize_optimization(self, solution, p, mode='IFM'):   
        solution = np.asarray(solution)
        rows, cols = solution.shape
        # 固定行优化流程
        fixed_rows = []
        if mode == 'IFM':
            data = np.ceil(solution).astype(int)
        else:
            # 初始化data数组
            data = np.empty((rows, cols), dtype=int)
            for col in range(cols):
                col_values = solution[:, col]
                # 按从小到大排序（便于找到第一个大于val的最小值）
                col_candidates = sorted(self.factors_candidate[col])
                for i in range(rows):
                    val = solution[i, col]
                    # 直接处理 val 为 1.0 的情况
                    if val == 1.0:
                        data[i, col] = 1  # 直接赋值为 1
                        continue  # 跳过当前循环，进入下一个
                    val = col_values[i]
                    # 查找第一个大于val的候选值，添加默认值避免异常
                    try:
                        data[i, col] = next(c for c in col_candidates if c >= val)
                    except StopIteration:
                        # 处理没有找到符合条件的值的情况
                        # 可以根据业务需求设置默认值，这里示例用最大值
                        data[i, col] = max(col_candidates) if col_candidates else 1

        
        # 从内层向外层调整, 把二维数组拉成一位数组，模仿maximize_row_with_constraint
        positions_to_adjust = np.where(data != 1)
        positions_to_adjust_list = list(zip(positions_to_adjust[0], positions_to_adjust[1]))
        columns_to_process = [r * cols + c for r, c in positions_to_adjust_list]
        best_data = data.copy()
        best_objective = 0
        iterations = [0]

        def constraint_func(data):
            # 除DRAM外每一个存储层次的存储容量约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_capacity = 0
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    break
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                for l in range(buffer_level+1):
                    for tensor in tensors_list:
                        tensor_capacity = 1
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim]
                        buffer_capacity += tensor_capacity
                # print("buffer_capacity: ", buffer_capacity)
                if len(tensors_list) > 0:
                    if buffer_capacity > self.buffer_size_list[buffer_key]:
                        return False
            
            # 并行容量约束
            for spatial_name in self.spatial_level:
                spatial_capacity = 1
                sp_level = self.spatial_level[spatial_name] 
                for c in range(cols):
                    spatial_capacity *= data[sp_level][c]
                # print("spatial_capacity: ", spatial_capacity)
                if spatial_capacity > self.spatial_size_list[spatial_name]:
                    return False
                
            return True

        def compute_column_upper_bound(data, original_value, columns_processed, row, col):
            test_data = np.ones((rows, cols))
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for cp in columns_processed:
                r = cp // cols
                c = cp % cols
                test_data[r][c] = data[r][c]
            
            low, high = 1, original_value
            best_valid = 1


            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)


        def bfs(current_data, col_index, columns_processed):
            iterations[0] += 1
            nonlocal best_data, best_objective

            if constraint_func(current_data):
                current_objective = np.sum(p * np.log2(current_data))
                # current_objective = np.sum(p_row * np.log2(current_data[row]) + a * np.log2(current_data[row])**2)
                if current_objective > best_objective:
                    best_data = current_data.copy()
                    best_objective = current_objective
                
            
            if col_index >= len(columns_to_process):
                return 

            actual_col = columns_to_process[col_index]
            row, col = actual_col // cols, actual_col % cols
            original_value = data[row][col]
            columns_processed_current = columns_processed.union({actual_col})
            # 计算上界并限制不超过原始值
            upper_bound = compute_column_upper_bound(current_data, original_value, columns_processed, row, col)

            if mode == 'IFM':
                values_to_try = range(upper_bound, 0, -1)
            else: # PFM
                values_to_try = [v for v in self.factors_candidate[col] if v <= upper_bound and v <= original_value]
            
            for value in values_to_try:
                # 保存原始值用于回溯
                old_value = current_data[row, col]
                current_data[row, col] = value
                
                # 递归处理下一列
                bfs(current_data, col_index + 1, columns_processed_current)
                
                # 回溯：恢复原始值
                current_data[row, col] = old_value

        bfs(data.copy(), 0, set())
        
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(best_data)


    def _integerize_with_staged_optimization5(self, solution, p, a):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            # low, high = 1, math.floor(original_value * 1.1)
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def maximize_row_with_constraint(data, row, constraint_func, target, fixed_rows, p_row):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data, best_objective
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    # current_objective = np.sum(p_row * current_data[row])
                    current_objective = np.sum(p_row * np.log2(current_data[row]) + a * np.log2(current_data[row])**2)
                    if current_objective > best_objective:
                        best_data = current_data.copy()
                        best_objective = current_objective
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        # 固定行优化流程
        fixed_rows = []
        data = np.ceil(solution).astype(int)
        
        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            p_row = p[sp_level]  # 获取当前行的权重
            data = maximize_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), 
                spatial_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
                
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            p_row = p[buffer_level]  # 获取当前行的权重
            constraint_func = create_buffer_constraint(buffer_name)
            data = maximize_row_with_constraint(
                data, buffer_level, 
                constraint_func, 
                buffer_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(buffer_level)
        
        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    def is_mapping_valid(self, mapping):
        factors = mapping.factor_dict
        key_order = 'RSPQCKHN'
        matrix = []

        # 遍历每个位置
        for i in range(len(self.targets)):
            row = []
            # 按照指定的键顺序遍历
            for key in key_order:
                row.append(factors[key][i])
            matrix.append(row)

        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = 0
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                break
            buffer_level = self.temporal_level[buffer_name]
            tensors_list = self.buffer_tensor_dict[buffer_name]
            for tensor in tensors_list:
                tmp = 1
                for l in range(buffer_level+1):
                    for dim in self.tensor_dimensions[tensor]:
                        tmp *= matrix[l][dim]
                buffer_capacity += tmp
            # print('buffer_capacity: ', buffer_capacity)
            if buffer_capacity > self.buffer_size_list[buffer_key]:
                return False
            
        for spatial_name in self.spatial_level:
            spatial_capacity = 1
            sp_level = self.spatial_level[spatial_name] 
            for c in range(8):
                spatial_capacity *= matrix[sp_level][c]
            # print('spatial_capacity: ', spatial_capacity)
            if spatial_capacity > self.spatial_size_list[spatial_name]:
                return False
            
        return True

    def create_genome(self, num_population, num_generations):
        mapping_list = []
        if self.mapper == "Cosa":
            prob_list = [copy.deepcopy(self.dimension_dict)]
            for i in range(num_population):
                new_prob = copy.deepcopy(self.dimension_dict)
                for key in new_prob.keys():
                    new_prob[key] = random.choice(self.expanded_dimension_dict[key])
                prob_list.append(new_prob)
            
            original_dir = os.getcwd()
            for i in range(len(prob_list)):
                operator_instance = copy.deepcopy(self.operator_instance)
                for key, value in prob_list[i].items():
                    if key in operator_instance:
                        operator_instance[key] = value
                problem = self.generate_prob_for_cosa(operator_instance)
                temp_dir = f'{original_dir}/cosa_tmp/temp_{uuid.uuid4()}'
                os.makedirs(temp_dir)
                utils.store_yaml(f'{temp_dir}/temp_prob.yaml', problem)
                utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
                utils.store_yaml(f'{temp_dir}/temp_mapspace.yaml', self.mapspace.mapspace_dict)
                succ = utils.run_cosa(temp_dir, f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_mapspace.yaml', f'{temp_dir}/temp_prob.yaml', cwd=temp_dir)
                map_path = f'{temp_dir}/temp_arch/map_16.yaml'
                
                if succ:
                    mapping = Mapping(map_path)
                    mapping_list.append(mapping)
                    map_path = f'{self.report_dir}/map.yaml'
                    utils.store_yaml(map_path, {"mapping": mapping.mapping})
                    prob_path = f'{self.report_dir}/problem.yaml'
                    utils.store_yaml(prob_path, self.problem.problem)
                    arch_path = f'{self.report_dir}/arch.yaml'
                    utils.store_yaml(arch_path, self.accelerator.arch_dict)

                    # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
                    utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)
                else:
                    print(False)
                    pass
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            # exit()


        elif self.mapper == "Soter":
            
            pass
        elif self.mapper == "MARS":
            start_time = time.time()
            if self.solver == 'lp':
                mapping = self.run_parameters2(num_population=num_population, num_generations=num_generations)
            elif self.solver == 'qp':
                mapping = self.run_parameters3(num_population=num_population, num_generations=num_generations)
            else:
                print("无效的选项，请选择 'lp' 或 'qp'。")
            print("耗时: ", time.time() - start_time)
            exit()
            for i in range(num_population):
                mapping_list.append(copy.deepcopy(mapping))
        elif self.mapper == "random":
            print("random")
            mapping = self.random_search()
            exit()
        else:
            print("无效的选项，请选择 'cosa'、'soter' 或 'MARS'。")
        return mapping_list

    def create_genome_for_parameters(self, num_population):
        para_list = []
        weight_matrix_len = 0
        if self.weight_matrix:
            for i in self.weight_matrix:
                para_list.append(np.array(i))
            weight_matrix_len = len(self.weight_matrix)
        for i in range(num_population-weight_matrix_len):
            if self.para_dim == 1:
                # p = []
                # for i in range(len(self.temporal_level) + len(self.spatial_level)):
                #     p.append(random.randint(0, 10))
                # para_list.append(p)
                row = len(self.temporal_level) + len(self.spatial_level) - 1
                para_list.append(np.random.randint(1, 11, size=row).astype(float))
            elif self.para_dim == 2:
                row = len(self.temporal_level) + len(self.spatial_level)
                col = 8
                # para_list.append(np.random.randint(1, 11, size=(row, col)).astype(float))
                para_list.append(np.random.uniform(low=1.0, high=100.0, size=(row, col)))
            else:
                print("无效的参数维度，请选择 1 或 2。")

        return para_list

    def select_parents(self, pop, fitness, num_parents, num_population, stage_idx=0, first_stage_value=None):
        # 选择在当前代中表现最好的个体作为下一代后代的父母
        if stage_idx==0:
            idx = np.argsort(fitness[:,0])[::-1]
            new_pop = [pop[i] for i in idx][:num_population]
            new_fitness = fitness[idx][:num_population]
            parents = copy.deepcopy(new_pop[:num_parents])
        else:
            elite_pop = [pop[i] for i in range(len(pop)) if all([fitness[i][kk]>=first_stage_value[kk] for kk in range(len(first_stage_value))])]
            elite_fitness = fitness
            for kk in range(len(first_stage_value)):
                elite_fitness = elite_fitness[(elite_fitness[:, kk] >= first_stage_value[kk])]
            idx = np.argsort(elite_fitness[:, stage_idx])[::-1]
            elite_pop = [elite_pop[i] for i in idx][:num_population]
            parents = copy.deepcopy(elite_pop[:num_parents])  if len(elite_pop)>0 else copy.deepcopy(pop[:num_parents])
            new_pop = pop[:num_population]
            new_fitness = fitness[:num_population]

        return new_pop, new_fitness, parents

    def crossover_tile(self, parents, pop, alpha=0.5):

        if len(parents) ==1:
            for idx in range(len(pop)):
                pop[idx] = copy.deepcopy(parents[0])
        else:
            for idx in range(0,len(pop),2):
                dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                # 为了保证交叉之后的正确性，目前只交换循环顺序，后面可以扩展交换factor
                for i in range(len(dad.permutation_list)):
                    if random.random() < alpha:
                        tmp = dad.permutation_list[i]
                        dad.permutation_list[i] = mom.permutation_list[i]
                        mom.permutation_list[i] = tmp
                for k in dad.factor_dict.keys():
                    if random.random() < alpha:
                        tmp = dad.factor_dict[k]
                        dad.factor_dict[k] = mom.factor_dict[k]
                        mom.factor_dict[k] = tmp
                pop[idx] = dad
                if idx + 1 < len(pop):
                    pop[idx+1] = mom

    def judge(self):
        try:
            values = []
            for term in self.fitness_obj:
                if term == "cycles":
                    reward = -self.observation['cycles']
                elif term == "energy":
                    reward = -self.observation['energy']
                elif term == "utilization":
                    reward = self.observation['utilization']
                elif term == "EDP":
                    reward = -self.observation['EDP']
                elif term == "GFLOPs":
                    reward = self.observation['GFLOPs']
                else:
                    raise NameError('Undefined fitness type')
                values.append(reward)
            return values
        except:
            return None
    
    
    def thread_fun(self, mapping):
        original_dir = os.getcwd()
        # 创建一个唯一的临时文件夹
        temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
        os.makedirs(temp_dir)
        try:
            if not self.is_mapping_valid(mapping):
                return None
        # try:
            remainders = {}
            outermost_idx = {}
            for d in mapping.factor_dict.keys():
                T = self.dimension_dict[d]
                F = mapping.factor_dict[d]
                remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
                
            
            temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders, outermost_idx)
            
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            utils.store_yaml(f'{temp_dir}/temp.yaml', temp_mapping)
            utils.store_yaml(f'{temp_dir}/temp_prob.yaml', self.problem.problem)
            utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
            utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp.yaml', cwd=temp_dir)
            try:
                self.observation = utils.parse_timeloop_output(output_stats_path)
            except:
                pass
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        return self.judge()

    def thread_fun_p(self, p):
        original_dir = os.getcwd()
        # 创建一个唯一的临时文件夹
        temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
        os.makedirs(temp_dir)
        try:
            if self.solver == 'lp':
                mapping, _ = self.generate_mapping(self.dimension, p)
            elif self.solver == 'qp':
                mapping, _ = self.generate_mapping(self.dimension, p[0], p[1])
            mapping = Mapping(mapping)
            remainders = {}
            outermost_idx = {}
            for d in mapping.factor_dict.keys():
                T = self.dimension_dict[d]
                F = mapping.factor_dict[d]
                remainders[d], outermost_idx[d] = utils.find_remainders2(F[::-1], T)
                
            
            temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders, outermost_idx)
            
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            utils.store_yaml(f'{temp_dir}/temp.yaml', temp_mapping)
            utils.store_yaml(f'{temp_dir}/temp_prob.yaml', self.problem.problem)
            utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
            utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp.yaml', cwd=temp_dir)
            try:
                self.observation = utils.parse_timeloop_output(output_stats_path)
            except:
                pass
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        return self.judge()

    def run(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        best_reward_list = []
        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None
        population = self.create_genome(num_population, num_generations)
        # reward_list = pool.map(self.thread_fun, population)
        # print(reward_list)

        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population

        for g in range(num_generations):
            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    print("restart!")
                    return 
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])

                self.crossover_tile(parents, population, alpha=0.57)
             
                # 变异
                for pop in population:
                    self.shuffle_factor_order(pop, alpha=0.5)
                    self.mutate_factor(pop, alpha=0.5)
                    self.mutate_permutation(pop, alpha=0.5)
                    self.mutate_dimemsion(pop, alpha=0.5)

                population = elite + population
                fitness = np.concatenate((elite_fitness, fitness))
                
                reward_list = pool.map(self.thread_fun, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness_obj
                }
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
        pool.close()

        remainders = {}
        outermost_idx = {}
        for d in best_sol.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_sol.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_sol.factor_dict, best_sol.permutation_list, best_sol.target_list, best_sol.type_list, best_sol.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        
        return chkpt
        

    def get_default_buffer_energy_cost(self):
        buf_energy_cost = {'DRAM': 200,
                           'l2': 2.2,
                           'l1': 1.12,
                           'MAC': 1.0,
        }
        return buf_energy_cost

    def get_expanded_problem(self):
        expanded_dict = {}
        for k in self.dimension_dict.keys():
            if self.dimension_dict[k] == 1 or self.dimension_dict[k] == 2 or (self.dimension_dict[k] & (self.dimension_dict[k]-1) == 0):
                expanded_dict[k] = [self.dimension_dict[k]]
            else:
                expanded_value = int(self.dimension_dict[k] * self.expanded_scope)
                if self.expanded_dict and k in self.expanded_dict:
                    expanded_list = list(set(range(expanded_value)) | set(self.expanded_dict[k]))
                else:
                    expanded_list = list(range(expanded_value))
                expanded_dict[k] = [num + self.dimension_dict[k] for num in expanded_list] if expanded_list else [self.dimension_dict[k]]
        return expanded_dict
    
    '''
    def run_parameters(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        def crossover_tile(parents, pop, alpha=0.5):
            if len(parents) ==1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0,len(pop),2):
                    dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                    dad = copy.deepcopy(dad)
                    mom = copy.deepcopy(mom)
                    # 为了保证交叉之后的正确性，目前只交换循环顺序，后面可以扩展交换factor
                    for i in range(len(dad)):
                        if random.random() < alpha:
                            tmp = dad[i]
                            dad[i] = mom[i]
                            mom[i] = tmp
                    pop[idx] = dad
                    if idx + 1 < len(pop):
                        pop[idx+1] = mom
        
        def shuffle_order(pop, alpha=0.5):
            if random.random() < alpha:
                return random.shuffle(pop)
            
        def mutate_factor(pop, alpha=0.5):
            for i in range(len(pop)):
                if random.random() < alpha:
                    if random.random() < alpha:
                        pop[i] = pop[i] * 2
                    else:
                        pop[i] = pop[i] / 2
        

        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        best_reward_list = []
        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None

        population = self.create_genome_for_parameters(num_population)
        
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population

        for g in range(num_generations):
            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    print("restart!")
                    return 
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])

                crossover_tile(parents, population, alpha=0.57)
             
                # 变异
                for pop in population:
                    shuffle_order(pop, alpha=0.5)
                    mutate_factor(pop, alpha=0.5)

                population = elite + population
                fitness = np.concatenate((elite_fitness, fitness))
                
                reward_list = pool.map(self.thread_fun_p, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness_obj
                }
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
        
        pool.close()

        remainders = {}
        outermost_idx = {}
        best_map, _ = self.generate_mapping(self.dimension, best_sol)
        best_map = Mapping(best_map)
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list, best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        return best_map
    '''

    def generate_candidate_solution(self, mode='IFM'):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            # print(tensor_capacity, data[l][dim])
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            # low, high = 1, math.floor(original_value * 1.1)
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def generate_row_with_constraint(data, row, constraint_func, target, fixed_rows):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    print(constraint_func(current_data), target, remaining_capacity_constraint(current_data, fixed_rows + [row]))
                    print(current_data)
                    best_data = current_data
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                if mode == 'IFM':
                    values_to_try = list(range(upper_bound, 0, -1))
                else: # PFM
                    values_to_try = [v for v in self.factors_candidate[actual_col] if v <= upper_bound and v <= original_value]
                
                random.shuffle(values_to_try)

                for value in values_to_try:
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        
        # 固定行优化流程
        fixed_rows = []
        matrix = []
        for row in range(self.num_rows):
            matrix_row = []
            for col in range(self.num_cols):
                if mode == 'IFM':
                    matrix_row.append(random.choice(self.candidates[row][col]))
                elif mode == 'PFM':
                    matrix_row.append(random.choice(self.factors_candidate[col]))
            matrix.append(matrix_row)
        data = np.array(matrix)

        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            data = generate_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), 
                spatial_capacity, 
                fixed_rows
            )
            
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            print(buffer_name)
            print("1:", self.temporal_level)
            print(self.tensor_dimensions)
            print(self.buffer_tensor_dict)
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
           
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            constraint_func = create_buffer_constraint(buffer_name)
            data = generate_row_with_constraint(
                data, buffer_level, 
                constraint_func, 
                buffer_capacity, 
                fixed_rows
            )
            fixed_rows.append(buffer_level)

        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    

    def random_search(self, max_iterations=10000, mode='IFM'):
        start_time = time.time()
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N
        """
        使用智能生成器进行随机搜索。
        """
        if mode == 'IFM':
            candidates = [[list(range(1, d+1)) for d in self.dimension] for _ in range(rows)]
        elif mode == 'PFM':
            candidates = [[utils.get_factors(d) for d in self.dimension] for _ in range(rows)]
        best_score = -float('inf')
        best_mapping = None
        valid_solutions_found = 0
        print(time.time() - start_time)
        print(f"开始优化的随机搜索，总迭代次数: {max_iterations}")
        original_dir = os.getcwd()
        # 创建一个唯一的临时文件夹
        temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
        os.makedirs(temp_dir)
        for i in range(max_iterations):
            
            print(time.time() - start_time)
            # 使用智能生成器生成候选解
            candidate_matrix = self.generate_candidate_solution()
            print(candidate_matrix)
            mapping_dict, _ = utils.generate_mapping_for_lpsolver2(candidate_matrix, self.targets, self.type, self.bypass)
            mapping = Mapping(mapping_dict)
            if self.is_mapping_valid(mapping):
                valid_solutions_found += 1

                remainders = {}
                outermost_idx = {}
                for d in mapping.factor_dict.keys():
                    T = self.dimension_dict[d]
                    F = mapping.factor_dict[d]
                    remainders[d], outermost_idx[d] = utils.find_remainders2(F[::-1], T)
                    
                
                temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders, outermost_idx)
                
                output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
                utils.store_yaml(f'{temp_dir}/temp.yaml', temp_mapping)
                utils.store_yaml(f'{temp_dir}/temp_prob.yaml', self.problem.problem)
                utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
                utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp.yaml', cwd=temp_dir)
                try:
                    self.observation = utils.parse_timeloop_output(output_stats_path)
                except:
                    pass
                current_score = self.judge()
                print("current_score", current_score)
                if current_score and current_score[0] > best_score:
                    best_score = current_score[0]
                    best_mapping = temp_mapping
                    print(f"迭代 {i+1}/{max_iterations}: 找到更优解，得分: {best_score}")
            else:
                continue

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        
        print(f"\n--- 搜索完成 ---")
        print(f"总尝试次数: {max_iterations}, 有效解数量: {valid_solutions_found}")
        print(f"有效解比例: {valid_solutions_found / max_iterations:.2%}")
        return best_mapping, best_score

    # 单目标
    def run_parameters(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        def crossover_tile(parents, pop, alpha=0.5):
            if len(parents) ==1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0, len(pop), 2):
                    dad = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    mom = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    
                    # 随机选择交叉策略
                    crossover_type = random.choice(['row_column', 'multi_point', 'arithmetic'])
                    
                    if crossover_type == 'row_column':
                        # 行/列交叉（原始实现）
                        if random.random() < alpha:
                            if dad.ndim == 2 and random.random() < 0.5:
                                # 行交叉
                                row_cutoff = random.randint(1, dad.shape[0]-1)
                                dad[row_cutoff:], mom[row_cutoff:] = mom[row_cutoff:], dad[row_cutoff:]
                            else:
                                # 列交叉
                                if dad.ndim == 1:
                                    col_cutoff = random.randint(1, len(dad)-1)
                                else:
                                    col_cutoff = random.randint(1, dad.shape[1]-1)
                                dad[:, col_cutoff:], mom[:, col_cutoff:] = mom[:, col_cutoff:], dad[:, col_cutoff:]
                    
                    elif crossover_type == 'multi_point':
                        # 多点交叉
                        if dad.ndim == 2:
                            rows, cols = dad.shape
                            for i in range(rows):
                                if random.random() < alpha:
                                    # 随机选择多个交叉点
                                    num_points = random.randint(1, cols//2)
                                    points = sorted(random.sample(range(1, cols), num_points))
                                    for j in range(len(points)):
                                        if j % 2 == 0:  # 交替交换片段
                                            start = points[j]
                                            end = points[j+1] if j+1 < len(points) else cols
                                            dad[i, start:end], mom[i, start:end] = mom[i, start:end], dad[i, start:end]
                        else:
                            if random.random() < alpha:
                                num_points = random.randint(1, len(dad)//2)
                                points = sorted(random.sample(range(1, len(dad)), num_points))
                                for j in range(len(points)):
                                    if j % 2 == 0:
                                        start = points[j]
                                        end = points[j+1] if j+1 < len(points) else len(dad)
                                        dad[start:end], mom[start:end] = mom[start:end], dad[start:end]
                    
                    elif crossover_type == 'arithmetic':
                        # 算术交叉：生成介于父代之间的子代
                        if dad.ndim == 2:
                            rows, cols = dad.shape
                            for i in range(rows):
                                for j in range(cols):
                                    if random.random() < alpha:
                                        beta = random.uniform(0.3, 0.7)  # 控制混合比例
                                        tmp = beta * dad[i, j] + (1-beta) * mom[i, j]
                                        mom[i, j] = beta * mom[i, j] + (1-beta) * dad[i, j]
                                        dad[i, j] = tmp
                        else:
                            for i in range(len(dad)):
                                if random.random() < alpha:
                                    beta = random.uniform(0.3, 0.7)
                                    tmp = beta * dad[i] + (1-beta) * mom[i]
                                    mom[i] = beta * mom[i] + (1-beta) * dad[i]
                                    dad[i] = tmp
                    
                    pop[idx] = dad
                    if idx + 1 < len(pop):
                        pop[idx+1] = mom

        
        def shuffle_order(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                if random.random() < alpha:
                    np.random.shuffle(arr)

            elif arr.ndim == 2:
                # 二维数组处理
                for row in arr:
                    if random.random() < alpha:
                        np.random.shuffle(row)

            else:
                raise ValueError("输入数组必须是一维或二维")
            
        def mutate_factor1(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < alpha:
                            arr[i] *= 2
                        else:
                            arr[i] *= 0.5
            elif arr.ndim == 2:
                # 二维数组处理
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            if random.random() < alpha:
                                arr[i][j] *= 2
                            else:
                                arr[i][j] *= 0.5
            else:
                raise ValueError("输入数组必须是一维或二维")

        def mutate_factor(arr, alpha=0.5, generation=0, max_generations=100):
            """带自适应幅度的因子变异"""
            # 变异幅度随代数衰减（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            # 随机选择变异方向（增大或减小）
                            if random.random() < 0.5:
                                # 增大因子（但幅度随迭代减小）
                                arr[i, j] *= (1 + 0.5 * decay_factor)
                            else:
                                # 减小因子
                                arr[i, j] *= (1 - 0.3 * decay_factor)
            else:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < 0.5:
                            arr[i] *= (1 + 0.5 * decay_factor)
                        else:
                            arr[i] *= (1 - 0.3 * decay_factor)
            
            return arr
 
        def gaussian_mutation(arr, alpha=0.5, generation=0, max_generations=100, sigma_base=0.2):
            """
            带自适应步长的高斯变异算子
            
            参数:
            - arr: 待变异的参数矩阵 (numpy array)
            - alpha: 变异概率 (0.0-1.0)
            - generation: 当前迭代代数
            - max_generations: 总迭代代数
            - sigma_base: 基础标准差，控制变异步长
            
            返回:
            - 变异后的参数矩阵
            """
            # 自适应标准差：随迭代进行减小步长（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            sigma = sigma_base * decay_factor
            
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:  # 以alpha概率进行变异
                            # 添加高斯噪声
                            arr[i, j] += np.random.normal(0, sigma)
                            
                            # 可选：限制参数范围（根据实际需求调整）
                            # arr[i, j] = np.clip(arr[i, j], 0, 1)  # 例如限制在[0,1]
            else:  # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] += np.random.normal(0, sigma)
                        # arr[i] = np.clip(arr[i], 0, 1)
            
            return arr

        def reverse_sign(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:  # 以alpha的概率反转符号
                        arr[i] *= -1  # 乘以-1实现符号反转
                return arr
            elif arr.ndim == 2:
                # 二维数组处理
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:  # 以alpha的概率反转符号
                            arr[i, j] *= -1  # 乘以-1实现符号反转
                return arr
            else:
                raise ValueError("输入数组必须是一维或二维")
        
        def shuffle_row(arr, alpha=0.5):
            if random.random() < alpha:
                rows = list(range(arr.shape[0]))
                random.shuffle(rows)
                arr[:] = arr[rows]
        
        def shuffle_col(arr, alpha=0.5):
            if random.random() < alpha:
                # print(arr.shape)
                cols = list(range(arr.shape[1]))
                # print(cols)
                random.shuffle(cols)
                arr[:] = arr[:, cols]

        def adaptive_crossover_rate(generation, max_generations, initial_rate=0.8):
            """自适应交叉率：随迭代减少探索"""
            progress = generation / max_generations
            return initial_rate * (1 - progress * 0.4)  # 从0.8降至0.4

        def adaptive_mutation_rate(generation, max_generations, initial_rate=0.5):
            """自适应变异率：随迭代增加开发"""
            progress = generation / max_generations
            return initial_rate * (0.5 + progress * 0.5)  # 从0.5降至0.25


        def adaptive_parameters(g, max_generations, initial_alpha=0.5):
            """根据迭代代数自适应调整参数"""
            # 在进化后期减少探索，增加开发
            progress = g / max_generations
            alpha = initial_alpha * (1 - progress * 0.5)  # 从0.5逐渐降低到0.25
            return alpha

        def check_convergence1(best_reward_list, window_size=10, threshold=1e-3):
            """检查是否收敛"""
            if len(best_reward_list) < window_size * 2:
                return False
            
            recent = np.array(best_reward_list[-window_size:])
            older = np.array(best_reward_list[-window_size*2:-window_size])
            
            # 计算相对变化
            relative_change = np.abs((np.mean(recent) - np.mean(older)) / np.mean(older))
            
            return relative_change < threshold

        def calculate_population_diversity(pop):
            """计算种群多样性（基于欧氏距离）"""
            if len(pop) <= 1:
                return 0
            diversity = 0
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):
                    diversity += np.linalg.norm(pop[i].flatten() - pop[j].flatten())
            return diversity / (len(pop) * (len(pop) - 1) / 2)
        
        def check_convergence(best_reward_list, window_size=10, threshold=1e-3):
            """改进的收敛检测"""
            if len(best_reward_list) < window_size * 2:
                return False
            recent = np.array([r[0] for r in best_reward_list[-window_size:]])
            older = np.array([r[0] for r in best_reward_list[-window_size*2:-window_size]])
            improvement = (np.mean(recent) - np.mean(older)) / (abs(np.mean(older)) + 1e-10)
            return improvement < threshold
        
        def simulated_annealing(solution, evaluate_func, para_dim, max_iter=100, initial_temp=100.0, cooling_rate=0.95):
            """
            模拟退火局部搜索函数
            
            参数:
            - solution: 当前解（参数矩阵）
            - evaluate_func: 评估函数，返回适应度值（单值，如stage_idx对应的奖励）
            - para_dim: 参数矩阵维度（1或2）
            - max_iter: 最大迭代次数
            - initial_temp: 初始温度
            - cooling_rate: 降温速率
            
            返回:
            - 优化后的解
            """
            current_sol = solution.copy()
            current_fitness = evaluate_func(current_sol)
            best_sol = current_sol.copy()
            best_fitness = current_fitness
            
            temp = initial_temp
            
            for t in range(max_iter):
                # 生成邻域解（根据维度选择变异方式）
                if para_dim == 1:
                    neighbor_sol = current_sol.copy()
                    idx = np.random.randint(len(neighbor_sol))
                    neighbor_sol[idx] += np.random.normal(0, 0.1)  # 高斯变异生成邻域解
                    neighbor_sol[idx] = np.clip(neighbor_sol[idx], 0.0, 1.0)  # 限制参数范围
                else:
                    neighbor_sol = current_sol.copy()
                    i, j = np.random.randint(0, neighbor_sol.shape[0]), np.random.randint(0, neighbor_sol.shape[1])
                    neighbor_sol[i, j] += np.random.normal(0, 0.1)
                    neighbor_sol[i, j] = np.clip(neighbor_sol[i, j], 0.0, 1.0)
                
                # 评估邻域解
                neighbor_fitness = evaluate_func(neighbor_sol)
                
                # 计算接受概率
                delta = neighbor_fitness - current_fitness
                if delta > 0 or np.random.rand() < math.exp(delta / temp):
                    current_sol = neighbor_sol
                    current_fitness = neighbor_fitness
                    
                    # 更新最优解
                    if current_fitness > best_fitness:
                        best_sol = current_sol.copy()
                        best_fitness = current_fitness
                
                # 降温
                temp *= cooling_rate
            
            return best_sol

        
        def evaluate_solution(solution):
            """辅助评估函数：返回当前解的适应度（单值）"""
            reward = self.thread_fun_p(solution)
            if reward is None or any(np.array(reward) >= 0):
                return -float("Inf")
            if stage_idx > 0 and any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                return -float("Inf")
            return reward[stage_idx]
    
        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        best_reward_list = []
        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None

        population = self.create_genome_for_parameters(num_population)
        # mapping, _ = self.generate_mapping(self.dimension, population[0])
        # exit()
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population
        for g in range(num_generations):
            start_time = time.time()
            # alpha = adaptive_parameters(g, num_generations)
            
            # 计算自适应参数
            diversity = calculate_population_diversity(population)
            crossover_alpha = adaptive_crossover_rate(g, num_generations)
            mutation_alpha = adaptive_mutation_rate(g, num_generations)

            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    print("restart!")
                    return 
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])

                crossover_tile(parents, population, alpha=crossover_alpha)
             
                # 变异
                for pop in population[num_elite:]:
                    # 选择一种主要变异策略
                    if random.random() < 0.7:  # 70%概率使用高斯变异
                        gaussian_mutation(pop, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    else:  # 30%概率使用因子变异
                        mutate_factor(pop, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    shuffle_order(pop, alpha=0.1)
                    # reverse_sign(pop, alpha=0.2)
                    if self.para_dim == 2:
                        shuffle_row(pop, alpha=0.1)
                        shuffle_col(pop, alpha=0.1)

                # 1. 更新种群（参数矩阵）
                population = elite + population[num_elite:]  # 合并精英和变异后的个体

                # 2. 更新适应度数组
                # 假设elite_fitness的长度是num_elite，population的长度是num_population
                # 需要确保新的fitness数组与population长度一致
                fitness = np.zeros((num_population, len(self.fitness_obj)))
                fitness[:num_elite] = elite_fitness  # 前num_elite个位置填充精英的适应度

                reward_list = pool.map(self.thread_fun_p, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness_obj
                }
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
                # # 检查收敛
                # if check_convergence(best_reward_list):
                #     print("Converged at generation", g)
                #     break
            
            # # 在迭代后期（如后半程）应用模拟退火
            # if g >= num_generations // 2:
            #     # 对当前最优解进行模拟退火优化
            #     if best_sol is not None:
            #         print(f"Generation {g}: 应用模拟退火优化最优解")
            #         best_sol = simulated_annealing(
            #             best_sol,
            #             evaluate_solution,
            #             para_dim=self.para_dim,
            #             max_iter=50,
            #             initial_temp=50.0,
            #             cooling_rate=0.98
            #         )
            #         # 重新评估优化后的解
            #         best_reward = [evaluate_solution(best_sol)] + [-float("Inf")] * (len(self.fitness_obj)-1)
            #         best_reward[stage_idx] = evaluate_solution(best_sol)
            
            # 每5代检查并恢复多样性
            if g % 5 == 0 and diversity < 0.2:
                # print(f"Generation {g}: 种群多样性低，注入新个体")
                replace_count = max(5, int(num_population * 0.1))
                for i in range(replace_count):
                    worst_idx = np.argmin(fitness[:, stage_idx])
                    population[worst_idx] = self.create_genome_for_parameters(1)[0]
                    fitness[worst_idx] = [float("-Inf")] * len(best_reward)
            elapsed_time = time.time() - start_time
            print("Generation {} 耗时: {:.3f}秒".format(g, elapsed_time))
        pool.close()

        remainders = {}
        outermost_idx = {}
        # print(best_sol)
        best_map, _ = self.generate_mapping(self.dimension, best_sol)
        best_map = Mapping(best_map)
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list, best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        return best_map        

    # 多目标
    def run_parameters2(self, num_population=10, num_generations=100, elite_ratio=0.05,
                   parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        def crossover_tile(parents, pop, alpha=0.5):
            if len(parents) ==1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0, len(pop), 2):
                    dad = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    mom = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    
                    # 随机选择交叉策略
                    crossover_type = random.choice(['row_column', 'multi_point', 'arithmetic'])
                    
                    if crossover_type == 'row_column':
                        # 行/列交叉（原始实现）
                        if random.random() < alpha:
                            if dad.ndim == 2 and random.random() < 0.5:
                                # 行交叉
                                row_cutoff = random.randint(1, dad.shape[0]-1)
                                dad[row_cutoff:], mom[row_cutoff:] = mom[row_cutoff:], dad[row_cutoff:]
                            else:
                                # 列交叉
                                if dad.ndim == 1:
                                    col_cutoff = random.randint(1, len(dad)-1)
                                else:
                                    col_cutoff = random.randint(1, dad.shape[1]-1)
                                dad[:, col_cutoff:], mom[:, col_cutoff:] = mom[:, col_cutoff:], dad[:, col_cutoff:]
                    
                    elif crossover_type == 'multi_point':
                        # 多点交叉
                        if dad.ndim == 2:
                            rows, cols = dad.shape
                            for i in range(rows):
                                if random.random() < alpha:
                                    # 随机选择多个交叉点
                                    num_points = random.randint(1, cols//2)
                                    points = sorted(random.sample(range(1, cols), num_points))
                                    for j in range(len(points)):
                                        if j % 2 == 0:  # 交替交换片段
                                            start = points[j]
                                            end = points[j+1] if j+1 < len(points) else cols
                                            dad[i, start:end], mom[i, start:end] = mom[i, start:end], dad[i, start:end]
                        else:
                            if random.random() < alpha:
                                num_points = random.randint(1, len(dad)//2)
                                points = sorted(random.sample(range(1, len(dad)), num_points))
                                for j in range(len(points)):
                                    if j % 2 == 0:
                                        start = points[j]
                                        end = points[j+1] if j+1 < len(points) else len(dad)
                                        dad[start:end], mom[start:end] = mom[start:end], dad[start:end]
                    
                    elif crossover_type == 'arithmetic':
                        # 算术交叉：生成介于父代之间的子代
                        if dad.ndim == 2:
                            rows, cols = dad.shape
                            for i in range(rows):
                                for j in range(cols):
                                    if random.random() < alpha:
                                        beta = random.uniform(0.3, 0.7)  # 控制混合比例
                                        tmp = beta * dad[i, j] + (1-beta) * mom[i, j]
                                        mom[i, j] = beta * mom[i, j] + (1-beta) * dad[i, j]
                                        dad[i, j] = tmp
                        else:
                            for i in range(len(dad)):
                                if random.random() < alpha:
                                    beta = random.uniform(0.3, 0.7)
                                    tmp = beta * dad[i] + (1-beta) * mom[i]
                                    mom[i] = beta * mom[i] + (1-beta) * dad[i]
                                    dad[i] = tmp
                    
                    pop[idx] = dad
                    if idx + 1 < len(pop):
                        pop[idx+1] = mom

        def shuffle_order(arr, alpha=0.5):
            if arr.ndim == 1:
                if random.random() < alpha:
                    np.random.shuffle(arr)
            elif arr.ndim == 2:
                for row in arr:
                    if random.random() < alpha:
                        np.random.shuffle(row)
            else:
                raise ValueError("输入数组必须是一维或二维")

        def mutate_factor1(arr, alpha=0.5):
            if arr.ndim == 1:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] *= 2 if random.random() < alpha else 0.5
            elif arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            arr[i][j] *= 2 if random.random() < alpha else 0.5
            else:
                raise ValueError("输入数组必须是一维或二维")

        def mutate_factor(arr, alpha=0.5, generation=0, max_generations=100):
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            if random.random() < 0.5:
                                arr[i, j] *= (1 + 0.5 * decay_factor)
                            else:
                                arr[i, j] *= (1 - 0.3 * decay_factor)
            else:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < 0.5:
                            arr[i] *= (1 + 0.5 * decay_factor)
                        else:
                            arr[i] *= (1 - 0.3 * decay_factor)
            return arr

        def gaussian_mutation(arr, alpha=0.5, generation=0, max_generations=100, sigma_base=0.2):
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            sigma = sigma_base * decay_factor
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            arr[i, j] += np.random.normal(0, sigma)
            else:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] += np.random.normal(0, sigma)
            return arr

        def reverse_sign(arr, alpha=0.5):
            if arr.ndim == 1:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] *= -1
                return arr
            elif arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            arr[i, j] *= -1
                return arr
            else:
                raise ValueError("输入数组必须是一维或二维")
        
        def shuffle_row(arr, alpha=0.5):
            if random.random() < alpha:
                rows = list(range(arr.shape[0]))
                random.shuffle(rows)
                arr[:] = arr[rows]

        def shuffle_col(arr, alpha=0.5):
            if random.random() < alpha:
                cols = list(range(arr.shape[1]))
                random.shuffle(cols)
                arr[:] = arr[:, cols]

        def adaptive_crossover_rate(generation, max_generations, initial_rate=0.8):
            progress = generation / max_generations
            return initial_rate * (1 - progress * 0.4)  # 从0.8降至0.4

        def adaptive_mutation_rate(generation, max_generations, initial_rate=0.5):
            progress = generation / max_generations
            return initial_rate * (0.5 + progress * 0.5)  # 变异率随代数调整

        # 以下两个函数用于多目标非支配排序及拥挤距离计算
        def non_dominated_sort(fitness):
            """
            计算非支配排序，返回若干个front，每个front为个体索引列表
            假设目标为均最大化。
            """
            population_size = fitness.shape[0]
            S = [set() for _ in range(population_size)]  # 每个个体支配的个体集合
            domination_count = [0] * population_size   # 每个个体被支配的次数
            fronts = [[]]

            for p in range(population_size):
                for q in range(population_size):
                    if p == q:
                        continue
                    # 如果p在所有目标上都不逊于q且至少在一个目标上更好，则认为p支配q
                    if all(fitness[p] >= fitness[q]) and any(fitness[p] > fitness[q]):
                        S[p].add(q)
                    elif all(fitness[q] >= fitness[p]) and any(fitness[q] > fitness[p]):
                        domination_count[p] += 1
                if domination_count[p] == 0:
                    fronts[0].append(p)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        domination_count[q] -= 1
                        if domination_count[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            fronts.pop()  # 最后一个为空
            return fronts

        def compute_crowding_distance(fitness, front):
            """
            计算指定front中每个个体的拥挤距离，返回一个字典：索引->距离
            """
            distance = {i: 0 for i in front}
            num_objectives = fitness.shape[1]
            for m in range(num_objectives):
                objs = [fitness[i, m] for i in front]
                sorted_indices = sorted(front, key=lambda i: fitness[i, m])
                f_min = fitness[sorted_indices[0], m]
                f_max = fitness[sorted_indices[-1], m]
                distance[sorted_indices[0]] = float('inf')
                distance[sorted_indices[-1]] = float('inf')
                if f_max - f_min == 0:
                    continue
                for k in range(1, len(sorted_indices)-1):
                    i = sorted_indices[k]
                    distance[i] += (fitness[sorted_indices[k+1], m] - fitness[sorted_indices[k-1], m]) / (f_max - f_min)
            return distance

        def select_parents_multi(population, fitness, num_parents):
            fronts = non_dominated_sort(fitness)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= num_parents:
                    selected.extend(front)
                else:
                    distances = compute_crowding_distance(fitness, front)
                    # 按拥挤度降序排列（拥挤度大的个体具有较好的多样性）
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    selected.extend(sorted_front[:num_parents - len(selected)])
                    break
            # 返回父代个体及其索引（后续可用于更新种群）
            selected_population = [population[i] for i in selected]
            return selected_population, selected

        # 模拟退火用于局部搜索
        def simulated_annealing(solution, evaluate_func, para_dim, max_iter=100, initial_temp=100.0, cooling_rate=0.95):
            current_sol = solution.copy()
            current_fitness = evaluate_func(current_sol)
            best_sol = current_sol.copy()
            best_fitness = current_fitness
            
            temp = initial_temp
            for t in range(max_iter):
                if para_dim == 1:
                    neighbor_sol = current_sol.copy()
                    idx = np.random.randint(len(neighbor_sol))
                    neighbor_sol[idx] += np.random.normal(0, 0.1)
                    neighbor_sol[idx] = np.clip(neighbor_sol[idx], 0.0, 1.0)
                else:
                    neighbor_sol = current_sol.copy()
                    i, j = np.random.randint(0, neighbor_sol.shape[0]), np.random.randint(0, neighbor_sol.shape[1])
                    neighbor_sol[i, j] += np.random.normal(0, 0.1)
                    neighbor_sol[i, j] = np.clip(neighbor_sol[i, j], 0.0, 1.0)
                neighbor_fitness = evaluate_func(neighbor_sol)
                # 若目标均为最大化，则delta为各目标差值之和（也可使用其他多目标聚合方式）
                delta = sum([n - c for n, c in zip(neighbor_fitness, current_fitness)])
                if delta > 0 or np.random.rand() < math.exp(delta / temp):
                    current_sol = neighbor_sol
                    current_fitness = neighbor_fitness
                    if sum(current_fitness) > sum(best_fitness):
                        best_sol = current_sol.copy()
                        best_fitness = current_fitness
                temp *= cooling_rate
            return best_sol

        # 多目标评估函数：直接返回目标向量
        def evaluate_solution(solution):
            reward = self.thread_fun_p(solution)
            # 若reward无效，则返回极差值（假设目标为最大化）
            if reward is None or any(np.array(reward) >= 0):
                return [float("-Inf") for _ in range(len(self.fitness_obj))]
            return reward

        # ----------------------------
        # 初始化种群和其他参数
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        # 用于归档非支配解，格式为：(solution, fitness)
        archive = []
        population = self.create_genome_for_parameters(num_population)
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population

        for g in range(num_generations):
            # start_time = time.time()   # 记录本轮迭代开始时间

            # 自适应参数
            diversity = 0
            # 计算种群多样性（基于欧氏距离），避免过早收敛
            if len(population) > 1:
                div_sum = 0
                count = 0
                for i in range(len(population)):
                    for j in range(i+1, len(population)):
                        div_sum += np.linalg.norm(population[i].flatten() - population[j].flatten())
                        count += 1
                diversity = div_sum / count
            crossover_alpha = adaptive_crossover_rate(g, num_generations)
            mutation_alpha = adaptive_mutation_rate(g, num_generations)
            finetune_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetune_iter):
                # 使用多目标选择父代
                parents, parent_indices = select_parents_multi(population, fitness, int(num_population * parents_ratio))
                if len(parents) < 1:
                    print("父代数量不足，重启")
                    return
                elite = copy.deepcopy(parents[:min(num_elite, len(parents))])
                # 对父代进行深复制保存适应度
                elite_fitness = [fitness[i].copy() for i in parent_indices[:min(num_elite, len(parents))]]

                # 交叉产生子代
                crossover_tile(parents, population, alpha=crossover_alpha)

                # 变异操作
                for pop_ind in population[num_elite:]:
                    if random.random() < 0.7:
                        gaussian_mutation(pop_ind, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    else:
                        mutate_factor(pop_ind, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    shuffle_order(pop_ind, alpha=0.1)
                    if self.para_dim == 2:
                        shuffle_row(pop_ind, alpha=0.1)
                        shuffle_col(pop_ind, alpha=0.1)
                # 合并精英与变异后的个体构成新种群
                population = elite + population[num_elite:]
                # 更新适应度
                fitness = np.zeros((num_population, len(self.fitness_obj)))
                # 精英的适应度复用
                for idx in range(len(elite)):
                    fitness[idx] = elite_fitness[idx]
                # 并行评估剩余个体
                reward_list = pool.map(self.thread_fun_p, population)
                for i in range(len(population)):
                    reward = reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf")] * len(self.fitness_obj)
                    fitness[i] = reward

                # 更新归档：结合历史归档和当前种群，取非支配前沿
                combined_solutions = [sol for sol, fit in archive] + population
                combined_fitness = np.concatenate([np.array([fit for sol, fit in archive]), fitness], axis=0) if archive else fitness

                fronts = non_dominated_sort(combined_fitness)

                # -----------------------------------------------------
                # # 选取第一前沿作为归档
                # archive = []
                # if fronts:
                #     for idx in fronts[0]:
                #         # 注意：如果idx超出当前种群大小，则源自归档
                #         if idx < len(archive):
                #             pass  # 此处仅做占位
                #     # 但简单起见，直接保存所有处于非支配前沿的解
                #     for idx in fronts[0]:
                #         archive.append( (combined_solutions[idx], combined_fitness[idx].copy()) )
                
                # 取第一前沿上的所有解
                candidate_indices = fronts[0] if fronts else []
                
                # 如果候选解数量超过 100，则利用拥挤距离进行筛选
                if len(candidate_indices) > 200:
                    distances = compute_crowding_distance(combined_fitness, candidate_indices)
                    # 根据拥挤距离降序排序，挑选拥挤度高的解（也可以选择其他排序依据）
                    sorted_indices = sorted(candidate_indices, key=lambda i: distances[i], reverse=True)
                    candidate_indices = sorted_indices[:200]
                
                # 清空原归档，并保存更新后的解
                archive = []
                for idx in candidate_indices:
                    archive.append((combined_solutions[idx], combined_fitness[idx].copy()))

                # -----------------------------------------------------

                # 父代数量逐步衰减
                num_parents = int(num_population * parents_ratio)
                parents_ratio *= ratio_decay

            # # 每5代检查种群多样性，若太低则注入新个体
            # if g % 5 == 0 and diversity < 0.2:
            #     replace_count = max(5, int(num_population * 0.1))
            #     for i in range(replace_count):
            #         worst_idx = np.argmin(fitness[:, 0])  # 可根据某个目标选择，也可混合指标
            #         population[worst_idx] = self.create_genome_for_parameters(1)[0]
            #         fitness[worst_idx] = [float("-Inf")] * len(self.fitness_obj)
            
            # # 可在后半程引入模拟退火进行局部搜索
            # if g >= num_generations // 2 and archive:
            #     # 选择归档中目标值和较均衡的个体（例如按总和最大）进行模拟退火改进
            #     best_idx = np.argmax([sum(fit) for sol, fit in archive])
            #     improved_sol = simulated_annealing(archive[best_idx][0],
            #                                     evaluate_solution,
            #                                     para_dim=self.para_dim,
            #                                     max_iter=50,
            #                                     initial_temp=50.0,
            #                                     cooling_rate=0.98)
            #     improved_fit = evaluate_solution(improved_sol)
            #     # 若改进后个体在归档中表现更好，则更新归档
            #     if sum(improved_fit) > sum(archive[best_idx][1]):
            #         archive[best_idx] = (improved_sol, improved_fit)
            
            # 计算并打印本轮迭代耗时
            # elapsed_time = time.time() - start_time
            # print("Generation {} 耗时: {:.3f}秒".format(g, elapsed_time))

        pool.close()

        # 最终从归档中选择一个折中解（例如：目标和最大的解）
        if archive:
            best_idx = np.argmax([sum(fit) for sol, fit in archive])
            best_sol = archive[best_idx][0]
            # print("Generation {} 当前最优解: {}".format(g, best_sol))
            # print("对应适应度: {}".format(fitness[best_idx]))
        else:
            best_sol = population[0]
            # print("Generation {} 当前最优解: {}".format(g, population[best_idx]))
            # print("对应适应度: {}".format(fitness[best_idx]))

        best_map, _ = self.generate_mapping(self.dimension, best_sol)
        best_map = Mapping(best_map)
        remainders = {}
        outermost_idx = {}
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list,
                                            best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)
        # arr_str  = np.array2string(best_sol, separator=', ')
        # print(arr_str)
        return best_map

    # 单目标, QP
    def run_parameters3(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        def crossover_tile(parents, pop, alpha=0.5):
            if len(parents) ==1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0, len(pop), 2):
                    dad = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    mom = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    
                    # 分别提取矩阵和候选参数
                    dad_matrix, dad_candidate = dad
                    mom_matrix, mom_candidate = mom
                    # 随机选择交叉策略
                    crossover_type = random.choice(['row_column', 'multi_point', 'arithmetic'])
                    
                    if crossover_type == 'row_column':
                        # 行/列交叉（原始实现）
                        if random.random() < alpha:
                            if dad_matrix.ndim == 2 and random.random() < 0.5:
                                # 行交叉
                                row_cutoff = random.randint(1, dad_matrix.shape[0]-1)
                                dad_matrix[row_cutoff:], mom_matrix[row_cutoff:] = mom_matrix[row_cutoff:], dad_matrix[row_cutoff:]
                            else:
                                # 列交叉
                                if dad_matrix.ndim == 1:
                                    col_cutoff = random.randint(1, len(dad_matrix)-1)
                                else:
                                    col_cutoff = random.randint(1, dad_matrix.shape[1]-1)
                                dad_matrix[:, col_cutoff:], mom_matrix[:, col_cutoff:] = mom_matrix[:, col_cutoff:], dad_matrix[:, col_cutoff:]
                    
                    elif crossover_type == 'multi_point':
                        # 多点交叉
                        if dad_matrix.ndim == 2:
                            rows, cols = dad_matrix.shape
                            for i in range(rows):
                                if random.random() < alpha:
                                    # 随机选择多个交叉点
                                    num_points = random.randint(1, cols//2)
                                    points = sorted(random.sample(range(1, cols), num_points))
                                    for j in range(len(points)):
                                        if j % 2 == 0:  # 交替交换片段
                                            start = points[j]
                                            end = points[j+1] if j+1 < len(points) else cols
                                            dad_matrix[i, start:end], mom_matrix[i, start:end] = mom_matrix[i, start:end], dad_matrix[i, start:end]
                        else:
                            if random.random() < alpha:
                                num_points = random.randint(1, len(dad_matrix)//2)
                                points = sorted(random.sample(range(1, len(dad_matrix)), num_points))
                                for j in range(len(points)):
                                    if j % 2 == 0:
                                        start = points[j]
                                        end = points[j+1] if j+1 < len(points) else len(dad_matrix)
                                        dad_matrix[start:end], mom_matrix[start:end] = mom_matrix[start:end], dad_matrix[start:end]
                    
                    elif crossover_type == 'arithmetic':
                        # 算术交叉：生成介于父代之间的子代
                        if dad_matrix.ndim == 2:
                            rows, cols = dad_matrix.shape
                            for i in range(rows):
                                for j in range(cols):
                                    if random.random() < alpha:
                                        beta = random.uniform(0.3, 0.7)  # 控制混合比例
                                        tmp = beta * dad_matrix[i, j] + (1-beta) * mom_matrix[i, j]
                                        mom_matrix[i, j] = beta * mom_matrix[i, j] + (1-beta) * dad_matrix[i, j]
                                        dad_matrix[i, j] = tmp
                        else:
                            for i in range(len(dad_matrix)):
                                if random.random() < alpha:
                                    beta = random.uniform(0.3, 0.7)
                                    tmp = beta * dad_matrix[i] + (1-beta) * mom_matrix[i]
                                    mom_matrix[i] = beta * mom_matrix[i] + (1-beta) * dad_matrix[i]
                                    dad_matrix[i] = tmp
                    
                    # 对候选参数部分采取交叉策略：
                    if random.random() < alpha:
                        child_dad_candidate = dad_candidate
                        child_mom_candidate = mom_candidate
                    else:
                        # 随机在两个父代候选中选择
                        child_dad_candidate = random.choice([dad_candidate, mom_candidate])
                        child_mom_candidate = random.choice([dad_candidate, mom_candidate])

                    pop[idx]   = (dad_matrix, child_dad_candidate)
                    if idx+1 < len(pop):
                        pop[idx+1] = (mom_matrix, child_mom_candidate)

        
        def shuffle_order(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                if random.random() < alpha:
                    np.random.shuffle(arr)

            elif arr.ndim == 2:
                # 二维数组处理
                for row in arr:
                    if random.random() < alpha:
                        np.random.shuffle(row)

            else:
                raise ValueError("输入数组必须是一维或二维")
            
        def mutate_factor1(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < alpha:
                            arr[i] *= 2
                        else:
                            arr[i] *= 0.5
            elif arr.ndim == 2:
                # 二维数组处理
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            if random.random() < alpha:
                                arr[i][j] *= 2
                            else:
                                arr[i][j] *= 0.5
            else:
                raise ValueError("输入数组必须是一维或二维")

        def mutate_factor(arr, alpha=0.5, generation=0, max_generations=100):
            """带自适应幅度的因子变异"""
            # 变异幅度随代数衰减（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:
                            # 随机选择变异方向（增大或减小）
                            if random.random() < 0.5:
                                # 增大因子（但幅度随迭代减小）
                                arr[i, j] *= (1 + 0.5 * decay_factor)
                            else:
                                # 减小因子
                                arr[i, j] *= (1 - 0.3 * decay_factor)
            else:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < 0.5:
                            arr[i] *= (1 + 0.5 * decay_factor)
                        else:
                            arr[i] *= (1 - 0.3 * decay_factor)
            
            return arr
 
        def gaussian_mutation(arr, alpha=0.5, generation=0, max_generations=100, sigma_base=0.2):
            """
            带自适应步长的高斯变异算子
            
            参数:
            - arr: 待变异的参数矩阵 (numpy array)
            - alpha: 变异概率 (0.0-1.0)
            - generation: 当前迭代代数
            - max_generations: 总迭代代数
            - sigma_base: 基础标准差，控制变异步长
            
            返回:
            - 变异后的参数矩阵
            """
            # 自适应标准差：随迭代进行减小步长（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            sigma = sigma_base * decay_factor
            
            if arr.ndim == 2:
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:  # 以alpha概率进行变异
                            # 添加高斯噪声
                            arr[i, j] += np.random.normal(0, sigma)
                            
                            # 可选：限制参数范围（根据实际需求调整）
                            # arr[i, j] = np.clip(arr[i, j], 0, 1)  # 例如限制在[0,1]
            else:  # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] += np.random.normal(0, sigma)
                        # arr[i] = np.clip(arr[i], 0, 1)
            
            return arr

        def reverse_sign(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:  # 以alpha的概率反转符号
                        arr[i] *= -1  # 乘以-1实现符号反转
                return arr
            elif arr.ndim == 2:
                # 二维数组处理
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:  # 以alpha的概率反转符号
                            arr[i, j] *= -1  # 乘以-1实现符号反转
                return arr
            else:
                raise ValueError("输入数组必须是一维或二维")
        
        def shuffle_row(arr, alpha=0.5):
            if random.random() < alpha:
                rows = list(range(arr.shape[0]))
                random.shuffle(rows)
                arr[:] = arr[rows]
        
        def shuffle_col(arr, alpha=0.5):
            if random.random() < alpha:
                # print(arr.shape)
                cols = list(range(arr.shape[1]))
                # print(cols)
                random.shuffle(cols)
                arr[:] = arr[:, cols]

        def adaptive_crossover_rate(generation, max_generations, initial_rate=0.8):
            """自适应交叉率：随迭代减少探索"""
            progress = generation / max_generations
            return initial_rate * (1 - progress * 0.4)  # 从0.8降至0.4

        def adaptive_mutation_rate(generation, max_generations, initial_rate=0.5):
            """自适应变异率：随迭代增加开发"""
            progress = generation / max_generations
            return initial_rate * (0.5 + progress * 0.5)  # 从0.5降至0.25


        def adaptive_parameters(g, max_generations, initial_alpha=0.5):
            """根据迭代代数自适应调整参数"""
            # 在进化后期减少探索，增加开发
            progress = g / max_generations
            alpha = initial_alpha * (1 - progress * 0.5)  # 从0.5逐渐降低到0.25
            return alpha

        def check_convergence1(best_reward_list, window_size=10, threshold=1e-3):
            """检查是否收敛"""
            if len(best_reward_list) < window_size * 2:
                return False
            
            recent = np.array(best_reward_list[-window_size:])
            older = np.array(best_reward_list[-window_size*2:-window_size])
            
            # 计算相对变化
            relative_change = np.abs((np.mean(recent) - np.mean(older)) / np.mean(older))
            
            return relative_change < threshold

        def calculate_population_diversity(pop):
            """计算种群多样性（基于欧氏距离）"""
            if len(pop) <= 1:
                return 0
            diversity = 0
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):
                    diversity += np.linalg.norm(pop[i][0].flatten() - pop[j][0].flatten())
            return diversity / (len(pop) * (len(pop) - 1) / 2)
        
        def check_convergence(best_reward_list, window_size=10, threshold=1e-3):
            """改进的收敛检测"""
            if len(best_reward_list) < window_size * 2:
                return False
            recent = np.array([r[0] for r in best_reward_list[-window_size:]])
            older = np.array([r[0] for r in best_reward_list[-window_size*2:-window_size]])
            improvement = (np.mean(recent) - np.mean(older)) / (abs(np.mean(older)) + 1e-10)
            return improvement < threshold
        
        def simulated_annealing(solution, evaluate_func, para_dim, max_iter=100, initial_temp=100.0, cooling_rate=0.95):
            """
            模拟退火局部搜索函数
            
            参数:
            - solution: 当前解（参数矩阵）
            - evaluate_func: 评估函数，返回适应度值（单值，如stage_idx对应的奖励）
            - para_dim: 参数矩阵维度（1或2）
            - max_iter: 最大迭代次数
            - initial_temp: 初始温度
            - cooling_rate: 降温速率
            
            返回:
            - 优化后的解
            """
            current_sol = solution.copy()
            current_fitness = evaluate_func(current_sol)
            best_sol = current_sol.copy()
            best_fitness = current_fitness
            
            temp = initial_temp
            
            for t in range(max_iter):
                # 生成邻域解（根据维度选择变异方式）
                if para_dim == 1:
                    neighbor_sol = current_sol.copy()
                    idx = np.random.randint(len(neighbor_sol))
                    neighbor_sol[idx] += np.random.normal(0, 0.1)  # 高斯变异生成邻域解
                    neighbor_sol[idx] = np.clip(neighbor_sol[idx], 0.0, 1.0)  # 限制参数范围
                else:
                    neighbor_sol = current_sol.copy()
                    i, j = np.random.randint(0, neighbor_sol.shape[0]), np.random.randint(0, neighbor_sol.shape[1])
                    neighbor_sol[i, j] += np.random.normal(0, 0.1)
                    neighbor_sol[i, j] = np.clip(neighbor_sol[i, j], 0.0, 1.0)
                
                # 评估邻域解
                neighbor_fitness = evaluate_func(neighbor_sol)
                
                # 计算接受概率
                delta = neighbor_fitness - current_fitness
                if delta > 0 or np.random.rand() < math.exp(delta / temp):
                    current_sol = neighbor_sol
                    current_fitness = neighbor_fitness
                    
                    # 更新最优解
                    if current_fitness > best_fitness:
                        best_sol = current_sol.copy()
                        best_fitness = current_fitness
                
                # 降温
                temp *= cooling_rate
            
            return best_sol

        
        def evaluate_solution(solution):
            """辅助评估函数：返回当前解的适应度（单值）"""
            reward = self.thread_fun_p(solution)
            if reward is None or any(np.array(reward) >= 0):
                return -float("Inf")
            if stage_idx > 0 and any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                return -float("Inf")
            return reward[stage_idx]
    
        def mutate_value(value, lower=0.5, upper=32, gauss_prob=0.8, sigma=1.6):
            """
            对一个实数进行变异。

            参数:
            value: 当前的实数值
            lower, upper: 取值范围的上下界
            gauss_prob: 采用高斯变异的概率（例如 0.8）
            sigma: 高斯噪声的标准差
            
            返回:
            变异后的实数，确保在[lower, upper]范围内
            """
            # 根据概率选择变异策略
            if random.random() < gauss_prob:
                # 高斯变异：在当前值上加上均值为0、标准差为sigma的噪声
                mutated_value = value + np.random.normal(0, sigma)
            else:
                # 均匀随机重置：直接从 [lower, upper] 区间内随机采样
                mutated_value = random.uniform(lower, upper)

            # 裁剪确保变异后数值在[lower, upper]范围内
            mutated_value = max(lower, min(upper, mutated_value))

            return mutated_value

        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        best_reward_list = []
        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None

        base_population = self.create_genome_for_parameters(num_population)
        # candidate_params = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64, 128, 256, 512, 1024]
        # population = [(genome, random.choice(candidate_params)) for genome in base_population]
        population = [(genome, random.uniform(0.5, 32)) for genome in base_population]
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population
        for g in range(num_generations):
            start_time = time.time()
            # alpha = adaptive_parameters(g, num_generations)
            
            # 计算自适应参数
            diversity = calculate_population_diversity(population)
            crossover_alpha = adaptive_crossover_rate(g, num_generations)
            mutation_alpha = adaptive_mutation_rate(g, num_generations)

            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    print("restart!")
                    return 
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])

                crossover_tile(parents, population, alpha=crossover_alpha)
             
                 # 变异：对精英之外的个体进行变异处理
                for i in range(num_elite, len(population)):
                    matrix, candidate = population[i]
                    # 对矩阵部分采用高斯变异或因子变异
                    if random.random() < 0.7:
                        matrix = gaussian_mutation(matrix, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    else:
                        matrix = mutate_factor(matrix, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    shuffle_order(matrix, alpha=0.1)
                    if self.para_dim == 2:
                        shuffle_row(matrix, alpha=0.1)
                        shuffle_col(matrix, alpha=0.1)
                    # 对候选参数部分以10%概率进行更新
                    # candidate = mutate_candidate(candidate, candidate_params, alpha=0.1)
                    candidate = mutate_value(candidate)
                    population[i] = (matrix, candidate)

                # 1. 更新种群（参数矩阵）
                population = elite + population[num_elite:]  # 合并精英和变异后的个体

                # 2. 更新适应度数组
                # 假设elite_fitness的长度是num_elite，population的长度是num_population
                # 需要确保新的fitness数组与population长度一致
                fitness = np.zeros((num_population, len(self.fitness_obj)))
                fitness[:num_elite] = elite_fitness  # 前num_elite个位置填充精英的适应度

                reward_list = pool.map(self.thread_fun_p, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness_obj
                }
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
                # # 检查收敛
                # if check_convergence(best_reward_list):
                #     print("Converged at generation", g)
                #     break
            
            # # 在迭代后期（如后半程）应用模拟退火
            # if g >= num_generations // 2:
            #     # 对当前最优解进行模拟退火优化
            #     if best_sol is not None:
            #         print(f"Generation {g}: 应用模拟退火优化最优解")
            #         best_sol = simulated_annealing(
            #             best_sol,
            #             evaluate_solution,
            #             para_dim=self.para_dim,
            #             max_iter=50,
            #             initial_temp=50.0,
            #             cooling_rate=0.98
            #         )
            #         # 重新评估优化后的解
            #         best_reward = [evaluate_solution(best_sol)] + [-float("Inf")] * (len(self.fitness_obj)-1)
            #         best_reward[stage_idx] = evaluate_solution(best_sol)
            
            # 每5代检查并恢复多样性
            if g % 5 == 0 and diversity < 0.2:
                # print(f"Generation {g}: 种群多样性低，注入新个体")
                replace_count = max(5, int(num_population * 0.1))
                for i in range(replace_count):
                    worst_idx = np.argmin(fitness[:, stage_idx])
                    population[worst_idx] = self.create_genome_for_parameters(1)[0]
                    fitness[worst_idx] = [float("-Inf")] * len(best_reward)
            elapsed_time = time.time() - start_time
            print("Generation {} 耗时: {:.3f}秒".format(g, elapsed_time))
        pool.close()

        remainders = {}
        outermost_idx = {}
        print(best_sol)
        best_map, _ = self.generate_mapping(self.dimension, best_sol[0], best_sol[1])
        best_map = Mapping(best_map)
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list, best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        return best_map        


    def generate_mapping_with_weight(self, weight):
        best_map, _ = self.generate_mapping(self.dimension, weight)
        best_map = Mapping(best_map)
        remainders = {}
        outermost_idx = {}
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list,
                                            best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)
        print("---------------------------------------")