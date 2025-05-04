from datetime import datetime
from cost_model import Timeloop
import pathlib
from input_objs import Arch, Prob, Mapspace, Mapping
import utils
import random
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD
import math

class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, expanded_scope, expanded_dict):
        self.opt_obj = [optim_obj, 'latency', 'energy']
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
        print(self.mapspace.buffer_tensor_dict)
        print("bypass: ", self.mapspace.bypass)
        print(self.tensor_dimensions)

        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels = self.accelerator.get_arch_info()

        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels

        print(operator_instance)
        print("buffer_name_list", self.buffer_name_list)
        print("buffer_size_list", self.buffer_size_list)
        print("buffer_spmap_cstr", self.buffer_spmap_cstr)
        print("buffers_with_spmap", self.buffers_with_spmap)
        print("num_buffer_level", self.num_buffer_level)
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        
        prob_path = pathlib.Path('{}/{}/{}/problem.yaml'.format(accelerator_dir, mapper, accelerator)).resolve()
        self.problem = Prob(prob_path)
        self.dimension, self.dimension_dict = self.problem.get_problem_info()
        self.expanded_dimension_dict = self.get_expanded_problem()
        print("dimension: ", self.dimension)
        print(self.dimension_dict)
        print(self.expanded_dimension_dict)

       
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

        print("temporal_level", self.temporal_level)
        print("spatial_level", self.spatial_level)
        print("obj_level", self.obj_level)
        print("spatial_size_list", self.spatial_size_list)
        print("targets", self.targets)
        print("type", self.type)

        mapping_dict, dimension_dict = self.generate_mapping(self.dimension)

        mapping = Mapping(mapping_dict)
        print(self.is_mapping_valid(mapping))




    def generate_mapping(self, dimension):
        sol = self.lpsolver(dimension)
        print(sol)
        mapping_dict, dimension_dict = utils.generate_mapping_for_lpsolver(sol, dimension, self.targets, self.type, self.bypass)

        return mapping_dict, dimension_dict

    def lpsolver(self, dimension_list):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

        # 定义目标函数：矩阵元素的加权和
        objective = 0
        for r in range(rows):
            obj_sum = 0
            for c in range(cols):
                level = self.obj_level[r]
                obj_sum += matrix[level][c]
            obj_sum *= rows - r
            objective += obj_sum

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

    def is_mapping_valid(self, mapping):
        factors = mapping.get_factors()
        key_order = 'RSPQCKHN'
        matrix = []

        # 遍历每个位置
        for i in range(len(self.targets)):
            row = []
            # 按照指定的键顺序遍历
            for key in key_order:
                row.append(factors[key][i])
            matrix.append(row)
        print(matrix)

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
            print("buffer_capacity: ", buffer_capacity)
            if buffer_capacity > self.buffer_size_list[buffer_key]:
                return False
            
        for spatial_name in self.spatial_level:
            spatial_capacity = 1
            sp_level = self.spatial_level[spatial_name] 
            for c in range(8):
                spatial_capacity *= matrix[sp_level][c]
            print("spatial_capacity: ", spatial_capacity)
            if spatial_capacity > self.spatial_size_list[spatial_name]:
                return False
            
        return True

    def create_genome(self, dimension_dict):
        if self.mapper == "Cosa":
            new_operator_instance = {}
            for key, value in self.operator_instance.items():
                if key == 'H':
                    pass 
                elif key in dimension_dict:
                    new_operator_instance[key] = dimension_dict[key]
                else:
                    new_operator_instance[key] = value

            print(new_operator_instance)
            prob_path = '../tmp/Cosa/problem.yaml'
            new_problem = {'problem': new_operator_instance}
            utils.generate_problem_for_cosa(prob_path, new_problem)
            utils.run_cosa(output_path='./tmp/Cosa/', prob_path=prob_path)
            pass
        elif self.mapper == "Soter":
            
            pass
        elif self.mapper == "MARS":
            
            pass
        else:
            print("无效的选项，请选择 'cosa'、'soter' 或 'MARS'。")
    
    def reinit_pop(self, num_population):
        for i in range(num_population):
            expanded_wokload = {}
            for key, values in self.expanded_dimension_dict.items():
                random_value = random.choice(values)
                expanded_wokload[key] = random_value
            self.create_genome(expanded_wokload, )
        pass
    
    def thread_fun(self):
        pass

    def run(self, epochs=10, num_population=20):
        self.reinit_pop(num_population)
        pass

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
            if self.dimension_dict[k] == 1 or self.dimension_dict[k] == 2:
                expanded_dict[k] = [self.dimension_dict[k]]
            else:
                expanded_value = int(self.dimension_dict[k] * self.expanded_scope)
                if self.expanded_dict and k in self.expanded_dict:
                    expanded_list = list(set(range(expanded_value)) | set(self.expanded_dict[k]))
                else:
                    expanded_list = list(range(expanded_value))
                expanded_dict[k] = [num + self.dimension_dict[k] for num in expanded_list] if expanded_list else [self.dimension_dict[k]]
        return expanded_dict

