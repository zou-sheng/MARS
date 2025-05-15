import yaml
import sys

def validate_yaml_config(config, file_path, required_keys=None):
    """验证YAML配置是否有效"""
    if config is None:
        print(f"错误: 文件 {file_path} 解析结果为空")
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                print(f"错误: 文件 {file_path} 缺少必要的键: {key}")
                return False
    
    return True

def calculate_buffer_accesses(mapping_config, problem_config):
    """计算各buffer的访问次数，考虑卷积参数和数据空间投影"""
    buffer_info = {
        'Registers': {'accesses': 0, 'kept_types': set()},
        'AccumulationBuffer': {'accesses': 0, 'kept_types': set()},
        'WeightBuffer': {'accesses': 0, 'kept_types': set()},
        'InputBuffer': {'accesses': 0, 'kept_types': set()},
        'GlobalBuffer': {'accesses': 0, 'kept_types': set()},
        'DRAM': {'accesses': 0, 'kept_types': {'Inputs', 'Outputs', 'Weights'}}
    }
    
    # 验证问题配置
    if not validate_yaml_config(problem_config, "problem.yaml", ['problem']):
        return buffer_info
    
    problem = problem_config.get('problem', {})
    
    # 提取问题实例参数
    instance = problem.get('instance', {})
    if not instance:
        print("错误: 问题配置中缺少'instance'键")
        return buffer_info
    
    # 提取数据空间配置
    shape = problem.get('shape', {})
    dimensions = shape.get('dimensions', [])
    data_projections = {}
    
    data_spaces = shape.get('data_spaces', [])
    for space in data_spaces:
        dtype = space.get('name')
        projection = space.get('projection', [])
        if dtype:
            data_projections[dtype] = projection
    
    # 初始维度大小（从problem配置中获取）
    initial_dimensions = {
        'H': instance.get('H', 1),
        'C': instance.get('C', 1),
        'K': instance.get('K', 1),
        'R': instance.get('R', 1),
        'S': instance.get('S', 1),
        'N': instance.get('N', 1),
        'P': instance.get('P', 1),
        'Q': instance.get('Q', 1),
        'Wdilation': instance.get('Wdilation', 1),
        'Wstride': instance.get('Wstride', 1),
        'Hdilation': instance.get('Hdilation', 1),
        'Hstride': instance.get('Hstride', 1),
    }
    
    current_dimensions = {
        'Inputs': initial_dimensions.copy(),
        'Outputs': initial_dimensions.copy(),
        'Weights': initial_dimensions.copy()
    }
    
    # 先处理datatype配置，设置各buffer保留的数据类型
    for i, config in enumerate(mapping_config):
        if not isinstance(config, dict):
            print(f"警告: 配置项 {i+1} 不是字典类型: {config}")
            continue
            
        if config.get('type') == 'datatype':
            target_buffer = config.get('target', '')
            kept_types = set(config.get('keep', []))
            
            if target_buffer in buffer_info:
                buffer_info[target_buffer]['kept_types'] = kept_types
    
    # 再处理temporal和spatial配置，计算访问次数
    for i, config in enumerate(mapping_config):
        if not isinstance(config, dict):
            print(f"警告: 配置项 {i+1} 不是字典类型: {config}")
            continue
            
        config_type = config.get('type', '')
        target_buffer = config.get('target', '')
        
        if config_type in ['temporal', 'spatial']:
            factors = config.get('factors', {})
            
            # 解析factors中的值
            parsed_factors = {}
            if isinstance(factors, str):
                for item in factors.split():
                    if '=' in item:
                        key, value = item.split('=')
                        # 处理带逗号的值，如"4,1"
                        if ',' in value:
                            parsed_factors[key] = [int(v) for v in value.split(',')]
                        else:
                            parsed_factors[key] = int(value)
            else:
                parsed_factors = factors
            
            # 应用映射因子
            for dtype in current_dimensions:
                for dim, factor in parsed_factors.items():
                    if dim in current_dimensions[dtype]:
                        # 对于列表形式的因子，取第一个值
                        if isinstance(factor, list):
                            factor = factor[0]
                        
                        current_dimensions[dtype][dim] = initial_dimensions[dim] // factor
            
            # 计算当前配置下的数据大小，考虑投影关系和卷积参数
            for dtype in current_dimensions:
                if dtype in buffer_info[target_buffer]['kept_types']:
                    tensor_size = 1
                    
                    # 根据数据类型的投影关系计算张量大小
                    if dtype in data_projections:
                        for proj_entry in data_projections[dtype]:
                            # 处理投影项可能是列表的情况
                            if isinstance(proj_entry, list):
                                # 获取主维度（列表的第一个元素）
                                if len(proj_entry) > 0:
                                    dim = proj_entry[0]
                                    
                                    # 确保dim是字符串类型
                                    if not isinstance(dim, str):
                                        print(f"警告: 投影维度不是字符串类型: {dim}")
                                        continue
                                    
                                    # 处理包含子维度的情况（如R -> Wdilation, P -> Wstride）
                                    if len(proj_entry) > 1 and isinstance(proj_entry[1], list):
                                        for sub_entry in proj_entry[1]:
                                            if isinstance(sub_entry, list) and len(sub_entry) > 0:
                                                sub_dim = sub_entry[0]
                                                if sub_dim in current_dimensions[dtype]:
                                                    # 例如：R × Wdilation 或 P × Wstride
                                                    if dim in current_dimensions[dtype]:
                                                        tensor_size *= current_dimensions[dtype][dim] * current_dimensions[dtype][sub_dim]
                                    else:
                                        if dim in current_dimensions[dtype]:
                                            tensor_size *= current_dimensions[dtype][dim]
                            else:
                                # 如果投影项不是列表，直接作为维度处理
                                dim = proj_entry
                                if isinstance(dim, str) and dim in current_dimensions[dtype]:
                                    tensor_size *= current_dimensions[dtype][dim]
                    else:
                        # 如果没有投影信息，使用所有维度
                        for dim in dimensions:
                            if dim in current_dimensions[dtype]:
                                tensor_size *= current_dimensions[dtype][dim]
                    
                    buffer_info[target_buffer]['accesses'] += tensor_size
    
    # 计算DRAM的访问次数
    for dtype in ['Inputs', 'Outputs', 'Weights']:
        tensor_size = 1
        
        # 根据数据类型的投影关系计算DRAM访问次数
        if dtype in data_projections:
            for proj_entry in data_projections[dtype]:
                # 处理投影项可能是列表的情况
                if isinstance(proj_entry, list):
                    if len(proj_entry) > 0:
                        dim = proj_entry[0]
                        
                        # 确保dim是字符串类型
                        if not isinstance(dim, str):
                            print(f"警告: 投影维度不是字符串类型: {dim}")
                            continue
                        
                        # 处理包含子维度的情况
                        if len(proj_entry) > 1 and isinstance(proj_entry[1], list):
                            for sub_entry in proj_entry[1]:
                                if isinstance(sub_entry, list) and len(sub_entry) > 0:
                                    sub_dim = sub_entry[0]
                                    if sub_dim in current_dimensions[dtype]:
                                        if dim in current_dimensions[dtype]:
                                            tensor_size *= current_dimensions[dtype][dim] * current_dimensions[dtype][sub_dim]
                        else:
                            if dim in current_dimensions[dtype]:
                                tensor_size *= current_dimensions[dtype][dim]
                else:
                    dim = proj_entry
                    if isinstance(dim, str) and dim in current_dimensions[dtype]:
                        tensor_size *= current_dimensions[dtype][dim]
        else:
            for dim in dimensions:
                if dim in current_dimensions[dtype]:
                    tensor_size *= current_dimensions[dtype][dim]
        
        buffer_info['DRAM']['accesses'] += tensor_size
    
    return {buffer: info['accesses'] for buffer, info in buffer_info.items()}

def print_mapping_summary(mapping_config):
    """打印映射配置摘要"""
    print("映射配置摘要:")
    for i, config in enumerate(mapping_config):
        if not isinstance(config, dict):
            print(f"错误: 配置项 {i+1} 格式不正确: {config}")
            continue
            
        config_type = config.get('type', '')
        target = config.get('target', '')
        
        if config_type in ['temporal', 'spatial']:
            factors = config.get('factors', {})
            permutation = config.get('permutation', '')
            
            # 解析factors中的值
            parsed_factors = {}
            if isinstance(factors, str):
                for item in factors.split():
                    if '=' in item:
                        key, value = item.split('=')
                        parsed_factors[key] = value
            else:
                parsed_factors = factors
            
            print(f"{i+1}. {config_type}映射 -> {target}:")
            print(f"   因子: {', '.join([f'{k}={v}' for k, v in parsed_factors.items()])}")
            print(f"   排列: {permutation}")
        
        elif config_type == 'datatype':
            kept = config.get('keep', [])
            bypassed = config.get('bypass', [])
            print(f"{i+1}. {config_type}配置 -> {target}:")
            print(f"   保留: {', '.join(kept)}")
            print(f"   旁路: {', '.join(bypassed)}")
        print()

def analyze_dataflow(mapping_config):
    """分析数据流向"""
    dataflow = {
        'Weights': [],
        'Inputs': [],
        'Outputs': []
    }
    
    # 初始化各buffer的保留和旁路配置
    buffer_config = {
        'Registers': {'keep': set(), 'bypass': set()},
        'AccumulationBuffer': {'keep': set(), 'bypass': set()},
        'WeightBuffer': {'keep': set(), 'bypass': set()},
        'InputBuffer': {'keep': set(), 'bypass': set()},
        'GlobalBuffer': {'keep': set(), 'bypass': set()},
        'DRAM': {'keep': {'Inputs', 'Outputs', 'Weights'}, 'bypass': set()}
    }
    
    # 提取datatype配置
    for i, config in enumerate(mapping_config):
        if not isinstance(config, dict):
            print(f"警告: 配置项 {i+1} 不是字典类型，跳过数据流向分析")
            continue
            
        if config.get('type') == 'datatype':
            target = config.get('target', '')
            if target in buffer_config:
                buffer_config[target]['keep'] = set(config.get('keep', []))
                buffer_config[target]['bypass'] = set(config.get('bypass', []))
    
    # 分析Weights流向
    weights_flow = ['DRAM']
    if 'Weights' in buffer_config['GlobalBuffer']['bypass']:
        weights_flow.append('GlobalBuffer(bypass)')
    else:
        weights_flow.append('GlobalBuffer(keep)')
    
    if 'Weights' in buffer_config['WeightBuffer']['keep']:
        weights_flow.append('WeightBuffer')
    
    if 'Weights' in buffer_config['Registers']['keep']:
        weights_flow.append('Registers')
    
    dataflow['Weights'] = weights_flow
    
    # 分析Inputs流向
    inputs_flow = ['DRAM']
    if 'Inputs' in buffer_config['GlobalBuffer']['keep']:
        inputs_flow.append('GlobalBuffer')
    
    if 'Inputs' in buffer_config['InputBuffer']['keep']:
        inputs_flow.append('InputBuffer')
    
    if 'Inputs' not in buffer_config['Registers']['bypass']:
        inputs_flow.append('Registers')
    
    dataflow['Inputs'] = inputs_flow
    
    # 分析Outputs流向
    outputs_flow = ['Registers']
    if 'Outputs' in buffer_config['AccumulationBuffer']['keep']:
        outputs_flow.append('AccumulationBuffer')
    
    if 'Outputs' in buffer_config['GlobalBuffer']['keep']:
        outputs_flow.append('GlobalBuffer')
    
    outputs_flow.append('DRAM')
    
    dataflow['Outputs'] = outputs_flow
    
    return dataflow

def main(mapping_file, problem_file):
    try:
        # 从YAML文件读取映射配置
        with open(mapping_file, 'r') as f:
            mapping_config = yaml.safe_load(f)
        
        # 验证映射配置
        if not validate_yaml_config(mapping_config, mapping_file):
            if isinstance(mapping_config, dict) and 'mapping' in mapping_config:
                print(f"检测到顶层键'mapping'，使用其值作为配置列表")
                mapping_config = mapping_config['mapping']
            else:
                sys.exit(1)
        
        # 从YAML文件读取问题配置
        with open(problem_file, 'r') as f:
            problem_config = yaml.safe_load(f)
        
        # 验证问题配置
        if not validate_yaml_config(problem_config, problem_file, ['problem']):
            sys.exit(1)
        
        print(f"已从 {mapping_file} 加载映射配置")
        print(f"已从 {problem_file} 加载问题配置\n")
        
        # 打印映射配置摘要
        print_mapping_summary(mapping_config)
        
        # 打印问题配置摘要
        print("问题配置摘要:")
        problem = problem_config.get('problem', {})
        instance = problem.get('instance', {})
        for key, value in instance.items():
            print(f"  {key}: {value}")
        print()
        
        # 计算各buffer的访问次数
        accesses = calculate_buffer_accesses(mapping_config, problem_config)
        
        # 分析数据流向
        dataflow = analyze_dataflow(mapping_config)
        
        # 打印数据流向
        print("\n数据流向分析:")
        for dtype, flow in dataflow.items():
            print(f"{dtype}: {' → '.join(flow)}")
        
        # 打印访问次数结果
        print("\n各Buffer访问次数计算结果:")
        total_accesses = sum(accesses.values())
        for buffer, count in sorted(accesses.items(), key=lambda x: -x[1]):
            percentage = (count / total_accesses) * 100
            print(f"{buffer}: {count:,} ({percentage:.2f}%)")
        
        print(f"\n总访问次数: {total_accesses:,}")
    
    except FileNotFoundError as e:
        print(f"错误: 文件不存在: {e.filename}")
    except yaml.YAMLError as e:
        print(f"YAML解析错误: {e}")
        print("请检查YAML文件格式是否正确")
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            print(f"错误位置: 第{mark.line+1}行, 第{mark.column+1}列")
    except Exception as e:
        print(f"发生未知错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python buffer_analysis.py <mapping.yaml> <problem.yaml>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
