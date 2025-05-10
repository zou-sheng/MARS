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

class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, expanded_scope, expanded_dict):
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
        print(sol)
        mapping_dict, dimension_dict = utils.generate_mapping_for_lpsolver(sol, dimension, self.targets, self.type, self.bypass)

        prob = copy.deepcopy(self.problem.problem)       
        for key in dimension_dict.keys():
            prob['problem']['instance'][key] = dimension_dict[key]
        return mapping_dict, prob

    def lpsolver(self, dimension_list, p=1):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

        # 定义目标函数：矩阵元素的加权和
        objective = 0
        # for r in range(rows):
        #     obj_sum = 0
        #     for c in range(cols):
        #         level = self.obj_level[r]
        #         obj_sum += matrix[level][c]
        #     obj_sum *= rows - r
        #     objective += obj_sum
        idx = 0
        for spatial_name in self.spatial_level:
            obj_sum = 0
            sp_level = self.spatial_level[spatial_name] 
            for c in range(cols):
                obj_sum += matrix[sp_level][c]
            obj_sum = obj_sum * (rows - idx) * p
            objective += obj_sum
            idx += 1

        # comp = math.prod(dimension_list)
        # ipt_data = math.prod()
        # comp_cycles = math.log(comp) - obj_sum

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
            obj_sum = obj_sum * (len(self.buffer_name_list.keys()) - idx) 
            objective += obj_sum
            idx += 1
        
        
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
        solution = [[2**matrix[i][j].value() for j in range(cols)] for i in range(rows)]

        return solution
    
    def lpsolver2(self, dimension_list, p=1):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

        # 定义目标函数：矩阵元素的加权和
        objective = 0
        # for r in range(rows):
        #     obj_sum = 0
        #     for c in range(cols):
        #         level = self.obj_level[r]
        #         obj_sum += matrix[level][c]
        #     obj_sum *= rows - r
        #     objective += obj_sum
        idx = 0
        for spatial_name in self.spatial_level:
            obj_sum = 0
            sp_level = self.spatial_level[spatial_name] 
            for c in range(cols):
                obj_sum += matrix[sp_level][c]
            obj_sum = obj_sum * (rows - idx) * p
            objective += obj_sum
            idx += 1

        # comp = math.prod(dimension_list)
        # ipt_data = math.prod()
        # comp_cycles = math.log(comp) - obj_sum

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
            obj_sum = obj_sum * (len(self.buffer_name_list.keys()) - idx) 
            objective += obj_sum
            idx += 1
        
        
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
        solution1 = [[2**matrix[i][j].value() for j in range(cols)] for i in range(rows)]

        solution = [[round(2**matrix[i][j].value()) for j in range(cols)] for i in range(rows)]
        print(solution1)
        solution = self._fix_constraints(solution, dimension_list)

        return solution

    def _fix_constraints(self, solution, dimension_list):
        # 修复维度乘积约束
        for c in range(len(solution[0])):
            current_prod = 1
            for r in range(len(solution)):
                current_prod *= solution[r][c]
            target = dimension_list[c]
            # 调整直到满足约束
            while not np.isclose(current_prod, target, rtol=0.1):
                for r in range(len(solution)):
                    original = solution[r][c]
                    delta = 1 if current_prod < target else -1
                    new_val = max(1, original + delta)
                    solution[r][c] = new_val
                    current_prod = current_prod // original * new_val
                    if np.isclose(current_prod, target, rtol=0.1):
                        break
        return solution

    def find_best_p(self):
        # 初始化迭代次数
        iterations = 10
        # 初始 p 值
        p_values = [1, 2]

        original_dir = os.getcwd()
        for _ in range(iterations):
            # 存储每个 p 值对应的目标函数值
            objective_values = []
            for p in p_values:
                mapping_dict, prob_dict = self.generate_mapping(self.dimension, p)
                mapping = Mapping(mapping_dict)
                
                temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
                os.makedirs(temp_dir)
                utils.store_yaml(f'{temp_dir}/temp_map.yaml', mapping)
                utils.store_yaml(f'{temp_dir}/temp_prob.yaml', prob_dict)
                utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
                utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp_map.yaml', cwd=temp_dir)
                output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
                try:
                    observation = utils.parse_timeloop_output(output_stats_path)
                    objective_values.append(observation['cycles'])
                except:
                    objective_values.append(float('inf'))
                
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            # 找到效果最好的 p 值的索引
            best_index = objective_values.index(min(objective_values))
            best_p = p_values[best_index]

            if best_p == p_values[0]:
                # 如果第一个 p 值效果好，下一次迭代比较 (p1 + p2) / 2 和 p1 - (p2 - p1) / 2
                new_p1 = (p_values[0] + p_values[1]) / 2
                new_p2 = p_values[0] - (p_values[1] - p_values[0]) / 2
                p_values = [new_p2, new_p1]
            else:
                # 如果第二个 p 值效果好，下一次迭代比较 (p1 + p2) / 2 和 p2 + (p2 - p1) / 2
                new_p1 = (p_values[0] + p_values[1]) / 2
                new_p2 = p_values[1] + (p_values[1] - p_values[0]) / 2
                p_values = [new_p1, new_p2]

        # 最后一次迭代后，再次评估两个 p 值，找到最终的最佳 p 值
        objective_values = []
        for p in p_values:
            mapping_dict, prob_dict = self.generate_mapping(self.dimension, p)
            mapping = Mapping(mapping_dict)
            temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
            os.makedirs(temp_dir)
            utils.store_yaml(f'{temp_dir}/temp_map.yaml', mapping)
            utils.store_yaml(f'{temp_dir}/temp_prob.yaml', prob_dict)
            utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
            utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp_map.yaml', cwd=temp_dir)
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            try:
                observation = utils.parse_timeloop_output(output_stats_path)
                objective_values.append(observation['cycles'])
            except:
                objective_values.append(float('inf'))
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        best_index = objective_values.index(min(objective_values))
        best_p = p_values[best_index]

        return best_p

    def find_best_p_2(self):
        # 初始 p 值
        p_values = [1,2,3,4,5,6,7,8,9,10]

        original_dir = os.getcwd()

        best_p = 1
        objective_values = float('inf')

        for p in p_values:
            mapping_dict, _ = self.generate_mapping(self.dimension, p)
            mapping = Mapping(mapping_dict) 
            remainders = {}
            for d in mapping.factor_dict.keys():
                T = self.dimension_dict[d]
                F = mapping.factor_dict[d]
                remainders[d] = utils.find_remainders(F[::-1], T)
            mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders)
            temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
            os.makedirs(temp_dir)
            utils.store_yaml(f'{temp_dir}/temp_map.yaml', mapping)
            utils.store_yaml(f'{temp_dir}/temp_prob.yaml', self.problem.problem)
            utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
            utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp_map.yaml', cwd=temp_dir)
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            try:
                observation = utils.parse_timeloop_output(output_stats_path)
                if observation['cycles'] < objective_values:
                    best_p = p
                    objective_values = observation['cycles']
                print(observation['cycles'])
            except:
                pass
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return best_p

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

    def create_genome(self, num_population):
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
            prob_list = [copy.deepcopy(self.dimension_dict)]
            for i in range(num_population):
                new_prob = copy.deepcopy(self.dimension_dict)
                for key in new_prob.keys():
                    new_prob[key] = random.choice(self.expanded_dimension_dict[key])
                prob_list.append(new_prob)

            # p = self.find_best_p_2()
            # print("best_p: ", p)
            # for i in range(len(prob_list)):
            #     dimension = [prob_list[i]['R'], prob_list[i]['S'], prob_list[i]['P'], prob_list[i]['Q'], prob_list[i]['C'], prob_list[i]['K'], prob_list[i]['H'], prob_list[i]['N']]
            #     mapping, dimension_dict = self.generate_mapping(dimension, p=32)
            #     for key in prob_list[i].keys():
            #         if key in dimension_dict:
            #             prob_list[i][key] = dimension_dict[key]
            #     mapping_list.append(Mapping(mapping))
            segment_count = 1
            p_values = [8, 0.5, 1, 2, 4, 8, 10]
            # 计算每段的长度
            segment_length = len(prob_list) // segment_count

            mapping_list = []
            for seg_idx in range(segment_count):
                # 确定当前分段的起始和结束索引
                start_idx = seg_idx * segment_length
                end_idx = start_idx + segment_length if seg_idx < segment_count - 1 else len(prob_list)

                # 获取当前分段对应的 p 值
                current_p = p_values[seg_idx]

                # 处理当前分段的 prob_list
                for i in range(start_idx, end_idx):
                    dimension = [prob_list[i]['R'], prob_list[i]['S'], prob_list[i]['P'], prob_list[i]['Q'], prob_list[i]['C'], prob_list[i]['K'], prob_list[i]['H'], prob_list[i]['N']]
                    mapping, dimension_dict = self.generate_mapping(dimension, p=current_p)
                    for key in prob_list[i].keys():
                        if key in dimension_dict:
                            prob_list[i][key] = dimension_dict[key]
                    mapping_list.append(Mapping(mapping))
                # exit()
        else:
            print("无效的选项，请选择 'cosa'、'soter' 或 'MARS'。")
        return mapping_list

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
            for d in mapping.factor_dict.keys():
                T = self.dimension_dict[d]
                F = mapping.factor_dict[d]
                remainders[d] = utils.find_remainders(F[::-1], T)
                
            
            temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders)
            
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

        population = self.create_genome(num_population)

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
        for d in best_sol.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_sol.factor_dict[d]
            remainders[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_sol.factor_dict, best_sol.permutation_list, best_sol.target_list, best_sol.type_list, best_sol.bypass_list, remainders)
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

