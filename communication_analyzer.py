import re
import argparse
from collections import defaultdict

# 架构专属数据流向规则（明确的层级链路）
ARCHITECTURE_FLOWS = {
    "Eyeriss": {
        "Weights": ["DRAM", "WeightRegFile", "__ARITH__"],  # DRAM→WeightRegFile→计算单元
        "Inputs": ["DRAM", "GlobalBuffer", "InputRegFile", "__ARITH__"],  # DRAM→GlobalBuffer→InputRegFile→计算单元
        "Outputs": ["DRAM", "GlobalBuffer", "PsumRegFile", "__ARITH__"]   # 双向：计算单元↔PsumRegFile→GlobalBuffer→DRAM
    },
    "Simba": {
        "Weights": ["DRAM", "WeightBuffer", "Registers", "__ARITH__"],  # DRAM→WeightBuffer→Registers→计算单元
        "Inputs": ["DRAM", "GlobalBuffer", "InputBuffer", "__ARITH__"],  # DRAM→GlobalBuffer→InputBuffer→计算单元
        "Outputs": ["DRAM", "GlobalBuffer", "AccumulationBuffer", "__ARITH__"]  # 双向：计算单元↔AccumulationBuffer→GlobalBuffer→DRAM
    }
}

def parse_report(report_path):
    """解析报告，提取模块的标量操作和能耗参数"""
    modules = defaultdict(dict)  # modules[模块名][数据类型][参数]
    current_module = None
    current_data_type = None  # Inputs/Weights/Outputs

    with open(report_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    for line in lines:
        # 匹配模块名称（如=== PsumRegFile ===）
        if line.startswith('===') and '===' in line:
            current_module = line.strip('= ').strip()
            current_data_type = None
            continue
        # 匹配数据类型子部分（Inputs/Weights/Outputs）
        if current_module and line.endswith(':'):
            data_type = line[:-1]
            if data_type in ["Inputs", "Weights", "Outputs"]:
                current_data_type = data_type
                modules[current_module][current_data_type] = {}
            continue
        # 提取标量操作和单位能耗
        if current_module and current_data_type:
            if 'Scalar reads (per-instance)' in line:
                reads = int(re.search(r':\s*(\d+)', line).group(1))
                modules[current_module][current_data_type]['scalar_reads'] = reads
            if 'Scalar fills (per-instance)' in line:
                fills = int(re.search(r':\s*(\d+)', line).group(1))
                modules[current_module][current_data_type]['scalar_fills'] = fills
            if 'Scalar updates (per-instance)' in line:
                updates = int(re.search(r':\s*(\d+)', line).group(1))
                modules[current_module][current_data_type]['scalar_updates'] = updates
            if 'Energy (per-scalar-access)' in line:
                energy_per = float(re.search(r':\s*(\d+\.\d+)', line).group(1))
                modules[current_module][current_data_type]['energy_per_access'] = energy_per
    return modules

def calculate_communication(modules, arch):
    """按明确的架构链路计算通信能耗"""
    total_energy = 0.0
    energy_details = []
    flows = ARCHITECTURE_FLOWS[arch]
    arithmetic = "__ARITH__"

    # --------------------------
    # 1. 输入链路（Weights/Inputs，单向从外层到计算单元）
    # --------------------------
    for data_type in ["Weights", "Inputs"]:
        chain = flows[data_type]
        # 遍历链路中的模块→模块环节（不含计算单元）
        for i in range(len(chain)-2):  # 最后一个是计算单元，不参与模块→模块
            outer_module = chain[i]
            inner_module = chain[i+1]
            if outer_module not in modules or inner_module not in modules:
                continue
            # 内层模块从外层模块填充数据（fills）
            inner_data = modules[inner_module].get(data_type, {})
            fills = inner_data.get('scalar_fills', 0)
            if fills == 0:
                continue
            # 外层模块的单位能耗
            outer_energy = modules[outer_module].get(data_type, {}).get('energy_per_access', 0)
            energy = fills * outer_energy
            energy_details.append(
                (f"{outer_module}→{inner_module}（{data_type}，输入链路）", 
                 fills, outer_energy, energy)
            )
            total_energy += energy

        # 最后一级模块→计算单元（读取）
        last_module = chain[-2]  # 计算单元前的最后一个模块
        if last_module in modules:
            module_data = modules[last_module].get(data_type, {})
            reads = module_data.get('scalar_reads', 0)  # 计算单元读取
            if reads > 0:
                energy_per = module_data.get('energy_per_access', 0)
                energy = reads * energy_per
                energy_details.append(
                    (f"{last_module}→{arithmetic}（{data_type}，输入到计算单元）", 
                     reads, energy_per, energy)
                )
                total_energy += energy

    # --------------------------
    # 2. 输出链路（Outputs，双向交互+回传）
    # --------------------------
    output_chain = flows["Outputs"]
    # 2.1 计算单元←→最后一级模块（双向）
    last_module = output_chain[-2]  # 计算单元前的最后一个模块（如PsumRegFile/AccumulationBuffer）
    if last_module in modules:
        module_data = modules[last_module].get("Outputs", {})
        # a. 计算单元读取（模块→计算单元）
        reads = module_data.get('scalar_reads', 0)
        if reads > 0:
            energy_per = module_data.get('energy_per_access', 0)
            energy = reads * energy_per
            energy_details.append(
                (f"{last_module}→{arithmetic}（Outputs，计算单元读取）", 
                 reads, energy_per, energy)
            )
            total_energy += energy
        # b. 计算单元写入（计算单元→模块）
        writes = module_data.get('scalar_updates', 0)
        if writes > 0:
            energy_per = module_data.get('energy_per_access', 0)
            energy = writes * energy_per
            energy_details.append(
                (f"{arithmetic}→{last_module}（Outputs，计算单元写入）", 
                 writes, energy_per, energy)
            )
            total_energy += energy

    # 2.2 输出回传（从最后一级模块逐层到DRAM）
    for i in range(len(output_chain)-2, 0, -1):  # 从计算单元前的模块反向回传
        inner_module = output_chain[i]
        outer_module = output_chain[i-1]
        if inner_module not in modules or outer_module not in modules:
            continue
        # 内层模块的更新次数 = 向外层模块回传的次数
        inner_data = modules[inner_module].get("Outputs", {})
        updates = inner_data.get('scalar_updates', 0)
        if updates == 0:
            continue
        # 内层模块的单位能耗（回传能耗）
        inner_energy = inner_data.get('energy_per_access', 0)
        energy = updates * inner_energy
        energy_details.append(
            (f"{inner_module}→{outer_module}（Outputs，结果回传）", 
             updates, inner_energy, energy)
        )
        total_energy += energy

    # 2.3 最终输出到DRAM（最外层模块→DRAM）
    if len(output_chain) >= 2 and "DRAM" in output_chain:
        final_module = output_chain[1]  # Outputs链路中DRAM的下一级模块（如GlobalBuffer）
        if final_module in modules and "DRAM" in modules:
            final_data = modules[final_module].get("Outputs", {})
            final_updates = final_data.get('scalar_updates', 0)
            if final_updates > 0:
                dram_energy = modules["DRAM"].get("Outputs", {}).get('energy_per_access', 0)
                energy = final_updates * dram_energy
                energy_details.append(
                    (f"{final_module}→DRAM（Outputs，最终输出）", 
                     final_updates, dram_energy, energy)
                )
                total_energy += energy

    return total_energy, energy_details

def main():
    parser = argparse.ArgumentParser(description='按明确架构链路计算通信能耗')
    parser.add_argument('report_path', help='报告文件路径')
    parser.add_argument('--arch', required=True, choices=['Eyeriss', 'Simba'], help='架构类型')
    args = parser.parse_args()

    modules = parse_report(args.report_path)
    total_energy, details = calculate_communication(modules, args.arch)

    # 输出结果
    # print(f"通信能耗计算详情（架构：{args.arch}，单位：pJ）：")
    # print("-" * 130)
    # print(f"{'传输环节':<60} | {'标量访问次数':<15} | {'单位能耗(pJ)':<15} | {'总能耗(pJ)':<20}")
    # print("-" * 130)
    # for item in details:
    #     print(f"{item[0]:<60} | {item[1]:<15} | {item[2]:<15.2f} | {item[3]:<20,.2f}")
    # print("-" * 130)
    # print(f"总通信能耗：{total_energy:.2f} pJ = {total_energy / 1e6:.4f} μJ")
    print(f"{total_energy / 1e6:.4f}")
if __name__ == "__main__":
    main()
