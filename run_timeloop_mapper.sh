for layer_id in {0..0}
do
    # 行号从1开始，layer_id=0对应第1行
    line_num=$((layer_id + 1))
    text=$(sed -n "${line_num}p" ./Benchmarks/deepbench_workload/layers.yaml)
    
    if [[ -n $text ]]; then
        filename=${text:2}
        
        # 确保输出目录存在
        output_dir=./timeloop_mapper/layer-${layer_id}/
        mkdir -p $output_dir
        
        timeloop-mapper ./test/arch.yaml ./test/mapspace.yaml ./test/mapper.yaml ./Benchmarks/deepbench_workload/${filename}.yaml -o $output_dir
        
        # if [[ $? -eq 0 ]]; then
        #     echo "第${layer_id}行处理完成，文件：${filename}"
        # else
        #     echo "命令执行失败，检查参数"
        # fi
    else
        echo "第${line_num}行为空，跳过处理"
    fi
done

timeloop-mapper ./test/arch.yaml ./test/mapspace.yaml ./test/mapper.yaml ./Benchmarks/deepbench_workload/01_DeepSpeech.yaml -o ./timeloop_mapper/layer-0/