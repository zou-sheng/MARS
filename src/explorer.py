from datetime import datetime
from cost_model import Timeloop
import pathlib
from input_objs import Arch, Prob

class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, type, report_dir, optim_obj, expanded_scope, expanded_dict):
        self.opt_obj = [optim_obj, 'latency', 'energy']
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.report_dir = report_dir 
        self.expanded_scope = expanded_scope
        self.expanded_dict = expanded_dict

        arch_path = pathlib.Path('{}/{}/{}.yaml'.format(accelerator_dir, accelerator, type)).resolve()
        self.accelerator = Arch(arch_path, accelerator)

        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes = self.accelerator.get_arch_info()

        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes

        print(self.buffer_name_list)
        print(self.buffer_size_list)
        print(self.buffer_spmap_cstr)
        print(self.buffers_with_spmap)
        print(self.num_buffer_level)
        print(self.num_pes)
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        
        prob_path = pathlib.Path('{}/{}/problem.yaml'.format(accelerator_dir, accelerator)).resolve()
        self.problem = Prob(prob_path)
        self.dimension, self.dimension_dict = self.problem.get_problem_info()
        self.expanded_dimension_dict = self.get_expanded_problem()
        print(self.dimension_dict)
        print(self.expanded_dimension_dict)

    def create_genome(self, dimension_dict):
        pass
    
    def reinit_pop(self, num_population):
        # 针对每个维度随机从扩展问题中选择一个值，然后针对这个值计算因数分解列表
        pass
    
    def thread_fun(self):
        pass

    def run(self, epochs=10, num_population=100):
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