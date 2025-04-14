from datetime import datetime
from cost_model import Timeloop

class MappingExplorer:
    def __init__(self, operator_instance, accelerator, report_dir, optim_obj, expanded_scope, expanded_dict):
        self.opt_obj = [optim_obj, 'latency', 'energy']
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.report_dir = report_dir 

        self.accelerator = accelerator
        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        # buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes = self.get_arch_info()

    def run(self, epochs):
        pass

    def get_arch_info(self):
        arch = copy.deepcopy(self.arch)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']