#!/bin/bash

# 定义根目录（根据实际情况修改，确保路径正确）
ROOT_DIR="/mars/report/RL/Simba/obj_EDP/deepbench_input1"

# 检查根目录是否存在
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误：根目录 $ROOT_DIR 不存在！"
    exit 1
fi

# 遍历所有 layer-* 文件夹
for layer_dir in "$ROOT_DIR"/layer-*/; do
    # 检查是否存在 expand-0.0 子目录
    target_dir="${layer_dir}expand-0.0/"
    if [ -d "$target_dir" ]; then
        echo "===== 处理目录: $target_dir ====="
        
        # 检查必要的 yaml 文件是否存在
        if [ -f "${target_dir}arch.yaml" ] && [ -f "${target_dir}map.yaml" ] && [ -f "${target_dir}problem.yaml" ]; then
            # 进入目标目录执行命令
            cd "$target_dir" || {
                echo "警告：无法进入目录 $target_dir，跳过"
                continue
            }
            
            # 执行 timeloop-model 命令
            echo "正在执行: timeloop-model arch.yaml map.yaml problem.yaml"
            timeloop-model arch.yaml map.yaml problem.yaml
            
            # 检查命令执行结果
            if [ $? -eq 0 ]; then
                echo "命令执行成功"
            else
                echo "警告：在 $target_dir 中执行命令失败"
            fi
            
            # 返回根目录，避免影响后续遍历
            cd "$ROOT_DIR" || exit 1
        else
            echo "警告：$target_dir 中缺少必要的 yaml 文件，跳过"
        fi
    else
        echo "跳过：$layer_dir 中未找到 expand-0.0 子目录"
    fi
done

echo "===== 所有目录处理完成 ====="