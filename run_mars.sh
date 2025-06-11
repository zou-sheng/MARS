cd ./src
start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
for layer_id in {56..56}
do
    printf "网络: deepbench, 层数: %d\n" $layer_id 
    python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
done
end_time=$(date +%s.%N)    # 获取结束时间
elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

echo "执行时间：${elapsed_time} 秒"

start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {0..511}
# do
#     printf "网络: conv, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Simba --mapper MARS --type arch --version v1 --workload conv --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# start_time=$(date +%s.%N)  # 获取开始时间（纳秒级时间戳，如：1689023456.123456789）
# for layer_id in {0..6}
# do
#     for batch_size in {1..64}
#     do
#         printf "网络: tensor, 层数: %d\n" $layer_id 
#         python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Simba --mapper MARS --type arch --version v1 --workload tensor --layer_id $layer_id --batch_size $batch_size --expanded_scope 0.1
#     done
# done
# end_time=$(date +%s.%N)    # 获取结束时间
# elapsed_time=$(echo "scale=3; $end_time - $start_time" | bc)  # 计算差值，保留3位小数（秒）

# echo "执行时间：${elapsed_time} 秒"

# for layer_id in {0..20}
# do
#     printf "网络: operators, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload operators --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..16}
# do
#     printf "网络: mobilenet, 层数: %d\n" $layer_id
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..22}
# do
#     printf "网络: unet, 层数: %d\n" $layer_id
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..53}
# do
#     printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..5}
# do
#     printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..7}
# do
#     printf "网络: alexnet, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..5}
# do
#     printf "网络: bertlarge, 层数: %d\n" $layer_id 
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator TensorCore --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
cd ..