import os

# 创建输出目录（如果不存在）
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# # 生成128个YAML文件，C从1到128
# for c in range(1, 129):
#     # 构建YAML内容
#     yaml_content = f"""problem:
#   C: {c}
#   Hdilation: 1
#   Hstride: 4
#   K: 64
#   N: 1
#   P: 55
#   Q: 55
#   R: 11
#   S: 11
#   Wdilation: 1
#   Wstride: 4
#   shape: cnn-layer
# """
    
#     # 定义文件名（例如：problem_C1.yaml, problem_C2.yaml, ...）
#     filename = os.path.join(output_dir, f"problem_C{c}.yaml")
    
#     # 写入文件
#     with open(filename, "w") as f:
#         f.write(yaml_content)
    
#     print(f"已生成文件: {filename}")

# 生成128个YAML文件，K从1到128
for k in range(1, 129):
    # 构建YAML内容
    yaml_content = f"""problem:
  C: 3
  Hdilation: 1
  Hstride: 4
  K: {k}
  N: 1
  P: 55
  Q: 55
  R: 11
  S: 11
  Wdilation: 1
  Wstride: 4
  shape: cnn-layer
"""
    
    # 定义文件名（例如：problem_C1.yaml, problem_C2.yaml, ...）
    filename = os.path.join(output_dir, f"problem_K{k}.yaml")
    
    # 写入文件
    with open(filename, "w") as f:
        f.write(yaml_content)
    
    print(f"已生成文件: {filename}")

# 生成128个YAML文件，P从1到128
for p in range(1, 129):
    # 构建YAML内容
    yaml_content = f"""problem:
  C: 3
  Hdilation: 1
  Hstride: 4
  K: 64
  N: 1
  P: {p}
  Q: 55
  R: 11
  S: 11
  Wdilation: 1
  Wstride: 4
  shape: cnn-layer
"""
    
    # 定义文件名（例如：problem_C1.yaml, problem_C2.yaml, ...）
    filename = os.path.join(output_dir, f"problem_P{p}.yaml")
    
    # 写入文件
    with open(filename, "w") as f:
        f.write(yaml_content)
    
    print(f"已生成文件: {filename}")

# 生成128个YAML文件，R从1到128
for r in range(1, 129):
    # 构建YAML内容
    yaml_content = f"""problem:
  C: 3
  Hdilation: 1
  Hstride: 4
  K: 64
  N: 1
  P: 55
  Q: 55
  R: {r}
  S: 11
  Wdilation: 1
  Wstride: 4
  shape: cnn-layer
"""
    
    # 定义文件名（例如：problem_C1.yaml, problem_C2.yaml, ...）
    filename = os.path.join(output_dir, f"problem_R{r}.yaml")
    
    # 写入文件
    with open(filename, "w") as f:
        f.write(yaml_content)
    
    print(f"已生成文件: {filename}")


print(f"共生成 {len(range(1, 129))} 个YAML文件到目录: {output_dir}")
