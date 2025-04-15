from datetime import datetime
from cost_model import Timeloop
import pathlib
from input_objs import Arch, Prob
import utils
import random

class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, report_dir, optim_obj, expanded_scope, expanded_dict):
        self.opt_obj = [optim_obj, 'latency', 'energy']
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.operator_instance = operator_instance
        self.report_dir = report_dir 
        self.expanded_scope = expanded_scope
        self.expanded_dict = expanded_dict
        self.mapper = mapper

        arch_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, type)).resolve()
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

        print(operator_instance)
        print(self.buffer_name_list)
        print(self.buffer_size_list)
        print(self.buffer_spmap_cstr)
        print(self.buffers_with_spmap)
        print(self.num_buffer_level)
        print(self.num_pes)
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        
        prob_path = pathlib.Path('{}/{}/{}/problem.yaml'.format(accelerator_dir, mapper, accelerator)).resolve()
        self.problem = Prob(prob_path)
        self.dimension, self.dimension_dict = self.problem.get_problem_info()
        self.expanded_dimension_dict = self.get_expanded_problem()
        print(self.dimension_dict)
        print(self.expanded_dimension_dict)

    def create_genome(self, dimension_dict):
        if self.mapper == "Random":
            
            pass
        elif self.mapper == "Cosa":
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
        elif self.mapper == "Mars":
            
            pass
        else:
            print("无效的选项，请选择 'random'、'cosa'、'soter' 或 'custom'。")
    
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