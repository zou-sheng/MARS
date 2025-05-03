import pathlib
import utils
import copy
import re
import numpy as np

class Arch:
    def __init__(self, arch_path, arch_name, version):
        self.path      = arch_path.resolve()
        self.arch_dict = utils.parse_yaml(self.path)
        self.arch_name = arch_name
        self.version   = version

    def config_str(self):
        """Return the filename for the input yaml with postfix."""
        return self.path.stem

    def print(self):
        print(self.__dict__)

    def get_arch_info(self):
        if self.version == 'v1':
            return self.get_arch_info_v1()
        elif self.version == 'v3':
            return self.get_arch_info_v3()
        else:
            raise Exception("不支持该版本！")

    def get_arch_info_v1(self):
        arch = copy.deepcopy(self.arch_dict)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['arch']

        num_arithmetic = arch['arithmetic']['instances']

        storage = arch['storage']
        num_buffer_levels = len(storage)


        for buffer in storage:
            buffer_name_list.append(buffer['name'])
            num_instances.append(buffer['instances'])
            if buffer['name'] == 'DRAM':
                buffer_size_list.append(float('Inf'))
            else:
                buffer_size_list.append(buffer['entries'])

        
        buffer_name_list.reverse()
        buffer_size_list.reverse()
        num_instances.reverse()
        num_instances.append(num_arithmetic)

        print(buffer_name_list, num_instances, buffer_size_list)

        # num_pes = int(num_arithmetic / reg_cluster_size)
        # assert(num_arithmetic % reg_cluster_size == 0)
        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)
       
        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels
        
    def get_arch_info_v3(self):
        arch = copy.deepcopy(self.arch_dict)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']

        if self.arch_name == 'Simba':
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search(r'.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            instances = int(macc.split('..')[1].split(']')[0]) + 1
            instances *= num_pes
            num_instances.append(instances)
        elif 'Eyeriss' in self.arch_name:
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            dummy_buffer = arch['subtree'][0]['subtree'][0]['local'][1]
            buffer_name = dummy_buffer['name']
            attributes = dummy_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search(r'.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            re_ret = re.search(r'.*\[', macc)
            if re_ret:
                instances = (int(macc.split('..')[1].split(']')[0]) + 1) * num_pes
            else:
                instances = num_pes
            num_instances.append(instances)
        elif 'TensorCore' in self.arch_name:
            main_memory = arch['subtree'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = main_memory['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = global_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            local_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = local_buffer['name']
            attributes = local_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = local_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = pe_buffer['name']
            attributes = pe_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = pe_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1
            num_pes = instances

            macc = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']
            re_ret = re.search(r'.*\[', macc)
            if re_ret:
                instances *= (int(macc.split('..')[1].split(']')[0]) + 1)
            num_instances.append(instances)

        print(buffer_name_list, num_instances, buffer_size_list)

        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)

        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels


class Prob:
    def __init__(self, prob_path):
        self.path = prob_path.resolve()
        self.problem = utils.parse_yaml(self.path)
        self.density = {'Inputs': 0.5, 'Weights': 1, 'Outputs': 1}
        self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}

    def get_problem_info(self):
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in self.dim2note.values():
            value = problem['problem']['instance'][key]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def config_str(self):
        """Returnsthe key str name for representing a unique layer."""
        val_arr = []
        for value in self.prob_bound:
            val_arr.append(str(value))
        keys = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']
        val_arr.extend([str(self.prob[key]) for key in keys])
        val_str = "_".join(val_arr)
        return val_str

    def print(self):
        print(self.__dict__)

class Mapping:
    def __init__(self, mapping_path):
        pass

class Mapspace:
    def __init__(self, mapspace_path):
        mapspace_dict = utils.parse_yaml(mapspace_path)
        self.mapspace = mapspace_dict['mapspace']

        for key, value in self.mapspace.items():
            setattr(self, key, value)

        self.var_idx_dict = {'Weights': 0, 'Inputs': 1, 'Outputs': 2}
        self.var_name_dict = {v: k for k, v in self.var_idx_dict.items()}
        self.org_idx_dict = {'spatial': 0, 'temporal': 1}
        self.org_name_dict = {v: k for k, v in self.org_idx_dict.items()}
        self.config_idx_dict = {'perm': 0, 'factor': 1}
        self.mapspace = None
        self.bypass = None
        self.arch = None
        self.prob = None
        self.factor_space = None
        self.factor_config_tup = None
        self.perm_space = None
        self.spatial_space = None


if __name__ == "__main__":
    arch_path = pathlib.Path('../SpatialAccelerators/Simba/arch_v1.yaml').resolve()
    simba = Arch(arch_path)