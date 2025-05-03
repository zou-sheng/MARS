import argparse
import yaml
import os
import pickle

from explorer import MappingExplorer

def main():
    opt             = parser.parse_args()
    benchmark_dir   = '../Benchmarks'
    accelerator_dir = '../SpatialAccelerators'
    config_dir      = '../config'
    accelerator     = opt.accelerator
    mapper          = opt.mapper
    type            = opt.type
    version         = opt.version
    workload        = opt.workload
    layer_id        = opt.layer_id
    expanded_scope  = opt.expanded_scope
    expanded_config = opt.expanded_config
    batch_size      = opt.batch_size
    
    with open(os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload)), 'r') as fd:
        layers = yaml.load(fd, Loader=yaml.SafeLoader)
    
    layer = layers[layer_id]
    
    report_dir = os.path.join(opt.report_dir, '{}'.format(mapper), '{}'.format(accelerator),'obj_{}'.format(opt.optim_obj),
                              '{}_input{}'.format(workload, batch_size), 'layer-{}'.format(layer_id), 'expand-{}'.format(expanded_scope))
   
    with open(os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(workload, layer)), 'r') as fd:
        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
        problem = {'problem': {
            'shape': {'name': 'CNN_Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                      'coefficients': [{'name': 'Wstride', 'default': 1},
                                       {'name': 'Hstride', 'default': 1},
                                       {'name': 'Wdilation', 'default': 1},
                                       {'name': 'Hdilation', 'default': 1}],
                      },
            'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                         'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                         }}}
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
            problem['problem']['shape']['data_spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]],
                               'read_write': True},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
            problem['problem']['instance']['type'] = 'T2D'
        else:
            problem['problem']['shape']['data_spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]]},
                              {'name': 'Outputs',
                               'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                               'read_write': True}]
            problem['problem']['instance']['type'] = 'C2D'
        if 'H' in layer_problem['problem'].keys():
            problem['problem']['instance']['H'] = layer_problem['problem']['H']
        else:
            problem['problem']['instance']['H'] = 1
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
            problem['problem']['instance']['N'] = layer_problem['problem']['N']
            problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
        else:
            problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
        problem['problem']['instance']['K'] = layer_problem['problem']['K']
        problem['problem']['instance']['C'] = layer_problem['problem']['C']
        problem['problem']['instance']['P'] = layer_problem['problem']['P']
        problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
        problem['problem']['instance']['R'] = layer_problem['problem']['R']
        problem['problem']['instance']['S'] = layer_problem['problem']['S']
        problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
        problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
        problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
        problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']
    
    with open(os.path.join(accelerator_dir, mapper, accelerator, 'problem.yaml'), 'w') as fd:
        yaml.dump(problem, fd)
    
    with open(os.path.join(config_dir, '{}.yaml'.format(expanded_config)), 'r') as fd:
        expanded_dict = yaml.load(fd, Loader=yaml.SafeLoader)

    ME = MappingExplorer(problem['problem']['instance'], accelerator_dir, accelerator, mapper, type, version, report_dir, opt.optim_obj, expanded_scope, expanded_dict)
    chkpt = ME.run(opt.epochs)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'env_chkpt.plt'), 'wb') as fd:
        pickle.dump(chkpt, fd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim_obj', type=str, default="latency", help='optimization objective')
    parser.add_argument('--epochs', type=int, default=10, help='number of generations/epochs')
    parser.add_argument('--report_dir', type=str, default='../report', help='The report directory')
    parser.add_argument('--accelerator', type=str, default='Simba', help='accelerator accelerator')
    parser.add_argument('--mapper', type=str, default='MARS', help='mapper tool')
    parser.add_argument('--type', type=str, default='arch', help='accelerator type')
    parser.add_argument('--version', type=str, default='v1', help='timeloop arch format version')
    parser.add_argument('--workload', type=str, default=None)
    parser.add_argument('--layer_id', type=int, default=None)
    parser.add_argument('--expanded_scope', type=float, default=1.0, help='expand the scope of the workload, range from 0 to 1')
    parser.add_argument('--expanded_config', type=str, default='expanded_config', help='The configuration file for the expansion of the workload')
    parser.add_argument('--batch_size', type=int, default=1)

    main()