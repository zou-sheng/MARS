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


class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, expanded_scope, expanded_dict, parameter_dimension=2):
        if optim_obj == 'latency':
            self.fitness_obj = ['cycles']
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.operator_instance = operator_instance
        
        self.report_dir = report_dir 
        self.expanded_scope = expanded_scope
        self.expanded_dict = expanded_dict
        self.mapper = mapper

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

    
    def generate_mapping(self, dimension, p):
        sol = self.lpsolver(dimension, p)
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
            prob += col_sum == math.log2(dimension_list[c])

        prob.solve(PULP_CBC_CMD(msg=0))
        # print(prob)
        
        # 记录开始时间
        start_time = time.time()
        solution = [[2**matrix[i][j].value() for j in range(cols)] for i in range(rows)]
        print("time1: ", time.time()-start_time)
        solution = self._integerize_with_staged_optimization(solution)
        print("time2: ", time.time()-start_time)
        return solution
    

    # --------------------- 优化后的maximize_row_with_constraint ---------------------
    def _maximize_row_with_constraint(self, data_np, row, constraint_func, target, objective_func, fixed_rows):
        """直接操作NumPy数组，避免数据格式转换"""
        row_data = data_np[row]
        # 找出原始值不为1的列（向量化筛选）
        non_one_mask = row_data != 1
        columns_to_process = np.flatnonzero(non_one_mask).astype(int)  # 直接获取列索引
        
        if columns_to_process.size == 0:
            return data_np.copy()  # 无列可处理，返回拷贝
        
        # 按值降序排列列索引（向量化排序）
        col_values = row_data[columns_to_process]
        sorted_indices = np.argsort(-col_values)
        columns_to_process = columns_to_process[sorted_indices]
        
        best_data = data_np.copy()
        best_objective = 0.0
        
        def bfs(col_index):
            nonlocal best_objective
            stack = [(0, best_data[row].copy())]  # (列索引, 当前行)
            
            while stack:
                col_index, current_row = stack.pop()
                
                # 检查约束条件
                if constraint_func() <= target and self._remaining_capacity_constraint(best_data, fixed_rows):
                    current_obj = objective_func()
                    if current_obj > best_objective:
                        best_objective = current_obj
                        best_data[:] = data_np
                
                if col_index >= len(columns_to_process):
                    continue
                    
                actual_col = columns_to_process[col_index]
                original_value = current_row[actual_col]
                
                # 从大到小尝试值
                for value in range(original_value, 0, -1):
                    new_row = current_row.copy()
                    new_row[actual_col] = value
                    data_np[row] = new_row
                    stack.append((col_index + 1, new_row.copy()))
            
            # 恢复原始行数据
            data_np[row] = best_data[row]

        
        bfs(0)
        return best_data


    # --------------------- 约束检查函数（复用向量化逻辑）---------------------
    def _remaining_capacity_constraint(self, data_np, fixed_rows_set):
        """使用集合进行固定行检查"""
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
                
            capacity = self._calculate_remaining_capacity_for_buffer(data_np, fixed_rows_set, buffer_name)
            if capacity > self.buffer_size_list[buffer_key]:
                return False
        
        return True
    
    def _calculate_remaining_capacity_for_buffer(self, data_np, fixed_rows_set, buffer_name):
        """计算单个buffer的容量"""
        buffer_level = self.temporal_level[buffer_name]
        tensors_list = self.buffer_tensor_dict[buffer_name]
        total_cap = 0
        
        for tensor in tensors_list:
            dims = self.tensor_dimensions[tensor]
            involved_rows = np.arange(buffer_level + 1)
            rows_mask = np.isin(involved_rows, list(fixed_rows_set))
            rows_data = np.where(rows_mask[:, None], data_np[involved_rows[:, None], dims], 1)
            total_cap += np.prod(rows_data, axis=(0, 1))
        
        return total_cap

    def _integerize_with_staged_optimization(self, solution):
        data_np = np.ceil(solution).astype(int)
        fixed_rows = set()  # 用集合加速查询
        # 缓存类属性
        buffer_name_list = self.buffer_name_list
        temporal_level = self.temporal_level
        buffer_tensor_dict = self.buffer_tensor_dict
        spatial_level = self.spatial_level
        spatial_size_list = self.spatial_size_list
        dimension = self.dimension

        # 预计算所有张量的维度信息
        tensor_dimensions_cache = {
            buffer_name: [self.tensor_dimensions[tensor] for tensor in tensors]
            for buffer_name, tensors in self.buffer_tensor_dict.items()
        }

        # --------------------- 空间层处理 ---------------------
        for spatial_name in spatial_level:
            sp_level = spatial_level[spatial_name]
            spatial_capacity = spatial_size_list[spatial_name]
            
            # 定义无参约束函数，直接捕获 data_np
            constraint_func = lambda: np.prod(data_np[sp_level])
            objective_func = lambda: np.prod(data_np[sp_level])
            
            data_np = self._maximize_row_with_constraint(
                data_np, sp_level,
                constraint_func=constraint_func,  # 无参函数
                target=spatial_capacity,
                objective_func=objective_func,  # 无参函数
                fixed_rows=fixed_rows
            )
            fixed_rows.add(sp_level)

        # --------------------- 缓冲层处理 ---------------------
        def vectorized_constraint(buffer_level, tensor_dims, fixed_rows_set):
            """向量化约束计算（闭包捕获fixed_rows_set）"""
            involved_rows = np.arange(buffer_level + 1)
            rows_mask = np.isin(involved_rows, list(fixed_rows_set))  # 转换为列表进行向量化判断
            rows_data = np.where(rows_mask[:, None], data_np[involved_rows[:, None], tensor_dims], 1)
            return np.sum(np.prod(rows_data, axis=(0, 1)))
        
        for buffer_key in sorted(buffer_name_list.keys()):
            buffer_name = buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                break
            buffer_level = temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            tensor_dims_list = tensor_dimensions_cache[buffer_name]
            
            def constraint_func(buffer_level=buffer_level, tensor_dims_list=tensor_dims_list):
                return sum(
                    vectorized_constraint(buffer_level, dims, fixed_rows)
                    for dims in tensor_dims_list
                )

            
            # 调用优化后的maximize_row_with_constraint
            data_np = self._maximize_row_with_constraint(
                data_np, buffer_level,
                constraint_func=lambda: constraint_func(),  # 包装为无参函数
                target=buffer_capacity,
                objective_func=constraint_func,  # 目标函数可复用约束函数（若目标为最小化约束）
                fixed_rows=fixed_rows
            )
            fixed_rows.add(buffer_level)
        
        # --------------------- 计算最后一行 ---------------------
        def compute_last_row():
            """原位计算最后一行（避免拷贝）"""
            if len(data_np) < 2:
                return data_np.tolist()
            # 计算前n-1行的乘积（跳过值为1的行，避免无效计算）
            product = np.prod(data_np[:-1], axis=0, where=data_np[:-1] != 1, initial=1)
            last_row = np.ceil(np.array(dimension) / product).astype(int)
            last_row = np.maximum(last_row, 1)
            data_np[-1] = last_row  # 原位修改
            return data_np.tolist()
        
        return compute_last_row()


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
                else:
                    print(False)
                    pass
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        elif self.mapper == "Soter":
            
            pass
        elif self.mapper == "MARS":
            mapping = self.run_parameters(num_population=num_population, num_generations=num_generations)
            exit()
            for i in range(num_population):
                mapping_list.append(copy.deepcopy(mapping))
        else:
            print("无效的选项，请选择 'cosa'、'soter' 或 'MARS'。")
        return mapping_list

    def create_genome_for_parameters(self, num_population):
        para_list = []
        for i in range(num_population):
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
                para_list.append(np.random.randint(3, 4, size=(row, col)).astype(float))
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
            mapping, _ = self.generate_mapping(self.dimension, p)
            mapping = Mapping(mapping)
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
            
        def mutate_factor(arr, alpha=0.5):
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
                                arr[i] *= 2
                            else:
                                arr[i] *= 0.5
            else:
                raise ValueError("输入数组必须是一维或二维")
            
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
                new_arr = arr[rows]
                for i in range(len(arr)):
                    arr[i] = new_arr[i]
        
        def shuffle_col(arr, alpha=0.5):
            if random.random() < alpha:
                cols = list(range(arr.shape[1]))
                random.shuffle(cols)
                new_arr = arr[cols]
                for i in range(len(arr)):
                    arr[i] = new_arr[i]

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
                    reverse_sign(pop, alpha=0.5)
                    if self.para_dim == 2:
                        shuffle_row(pop, alpha=0.5)
                        shuffle_col(pop, alpha=0.5)

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

