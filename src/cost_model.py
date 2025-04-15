import os
import yaml

class Timeloop:
    def __init__(self, in_config_path='../SpatialAccelerators', out_config_path='../out_config', accelerator='Simba',
                 opt_obj=None, use_sparse=False):
        self.accelerator     = accelerator
        self.out_config_path = out_config_path
        self.use_sparse      = use_sparse
        with open(os.path.join(in_config_path, accelerator, 'arch.yaml'), 'r') as fd:
            self.arch = yaml.load(fd, Loader=yaml.SafeLoader)
   
        with open(os.path.join(in_config_path, accelerator, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd,Loader=yaml.SafeLoader)

        with open(os.path.join(in_config_path, accelerator, 'mapspace.yaml'), 'r') as fd:
            self.mapspace = yaml.load(fd, Loader=yaml.SafeLoader)

        self.opt_obj = opt_obj

        if self.use_sparse:
            with open(os.path.join(in_config_path, accelerator, 'sparse.yaml'), 'r') as fd:
                self.sparse = yaml.load(fd,Loader=yaml.SafeLoader)

        self._executable = 'timeloop-model'

        pass