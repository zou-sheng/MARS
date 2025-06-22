import os
import yaml

# 确保中文正常显示
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_architecture(a, b):
    """
    根据给定的a和b值生成架构配置
    
    参数:
    a (int): 水平维度大小
    b (int): 垂直维度大小
    """
    return {
        'arch': {
            'arithmetic': {
                'instances': a * b,
                'word-bits': 8
            },
            'storage': [
                {
                    'name': 'Registers',
                    'entries': 1,
                    'instances': a * b,
                    'word-bits': 8,
                    'cluster-size': b,
                    'num-ports': 2,
                    'num-banks': 8
                },
                {
                    'name': 'AccumulationBuffer',
                    'entries': 3072,
                    'instances': a,
                    'word-bits': 24,
                    'cluster-size': 1,
                    'network-word-bits': 16,
                    'num-ports': 2,
                    'num-banks': 2
                },
                {
                    'name': 'WeightBuffer',
                    'entries': 32768,
                    'instances': a,
                    'word-bits': 8,
                    'block-size': 4,
                    'num-ports': 1,
                    'num-banks': 8
                },
                {
                    'name': 'InputBuffer',
                    'entries': 8192,
                    'instances': a,
                    'word-bits': 8,
                    'block-size': 4,
                    'num-ports': 2,
                    'num-banks': 1
                },
                {
                    'name': 'GlobalBuffer',
                    'entries': 65536,
                    'instances': 1,
                    'word-bits': 8,
                    'block-size': 8,
                    'num-ports': 2,
                    'num-banks': 256
                },
                {
                    'name': 'DRAM',
                    'technology': 'DRAM',
                    'instances': 1,
                    'word-bits': 8,
                    'block-size': 64,
                    'bandwidth': 1
                }
            ]
        }
    }

# MapSpace配置数据
mapspace_config = {
    'mapspace': {
        'constraints': [
            {
                'target': 'Registers',
                'type': 'datatype',
                'keep': ['Weights'],
                'bypass': ['Inputs', 'Outputs']
            },
            {
                'target': 'AccumulationBuffer',
                'type': 'datatype',
                'keep': ['Outputs'],
                'bypass': ['Weights', 'Inputs']
            },
            {
                'target': 'WeightBuffer',
                'type': 'datatype',
                'keep': ['Weights'],
                'bypass': ['Inputs', 'Outputs']
            },
            {
                'target': 'InputBuffer',
                'type': 'datatype',
                'keep': ['Inputs'],
                'bypass': ['Weights', 'Outputs']
            },
            {
                'target': 'GlobalBuffer',
                'type': 'datatype',
                'keep': ['Inputs', 'Outputs'],
                'bypass': ['Weights']
            },
            {
                'target': 'DRAM',
                'type': 'datatype',
                'keep': ['Inputs', 'Outputs', 'Weights']
            }
        ]
    }
}

# Problem配置数据
problem_config = {
    'problem': {
        'instance': {
            'C': 3,
            'H': 224,
            'Hdilation': 1,
            'Hstride': 1,
            'K': 63,
            'N': 224,
            'P': 217,
            'Q': 217,
            'R': 7,
            'S': 7,
            'Wdilation': 1,
            'Wstride': 2,
            'type': 'C2D'
        },
        'shape': {
            'coefficients': [
                {'default': 1, 'name': 'Wstride'},
                {'default': 1, 'name': 'Hstride'},
                {'default': 1, 'name': 'Wdilation'},
                {'default': 1, 'name': 'Hdilation'}
            ],
            'data-spaces': [
                {
                    'name': 'Weights',
                    'projection': [
                        [['H']],
                        [['C']],
                        [['K']],
                        [['R']],
                        [['S']]
                    ]
                },
                {
                    'name': 'Inputs',
                    'projection': [
                        [['N']],
                        [['H']],
                        [['C']],
                        [['R', ['Wdilation']], ['P', ['Wstride']]],
                        [['S', ['Hdilation']], ['Q', ['Hstride']]]
                    ]
                },
                {
                    'name': 'Outputs',
                    'projection': [
                        [['N']],
                        [['H']],
                        [['K']],
                        [['Q']],
                        [['P']]
                    ],
                    'read-write': True
                }
            ],
            'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
            'name': 'CNN-Layer'
        }
    }
}

def main():
    """
    主函数：生成从16×64到64×64的所有架构配置文件
    """
    # 创建输出目录
    main_output_dir = "./"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 遍历a从16到64，b固定为64
    for a in range(1, 65):  # 范围是[16, 64]，包含64
        b = 64
        # 生成架构配置
        config = generate_architecture(a, b)
        # 为当前配置创建单独的子目录
        config_dir = os.path.join(main_output_dir, f"Simba_{a}x{b}")
        os.makedirs(config_dir, exist_ok=True)
        
        # 保存架构配置到arch.yaml
        arch_filename = os.path.join(config_dir, f"arch.yaml")
        with open(arch_filename, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        # 保存MapSpace配置到mapspace.yaml
        mapspace_filename = os.path.join(config_dir, f"mapspace.yaml")
        with open(mapspace_filename, 'w', encoding='utf-8') as f:
            yaml.dump(mapspace_config, f, allow_unicode=True, default_flow_style=False)
        
        # 保存Problem配置到problem.yaml
        problem_filename = os.path.join(config_dir, f"problem.yaml")
        with open(problem_filename, 'w', encoding='utf-8') as f:
            yaml.dump(problem_config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"已生成配置文件: {arch_filename}")
        print(f"已生成配置文件: {mapspace_filename}")
        print(f"已生成配置文件: {problem_filename}")
    
    print(f"\n所有配置文件已生成到 {main_output_dir} 目录")

if __name__ == "__main__":
    main()
