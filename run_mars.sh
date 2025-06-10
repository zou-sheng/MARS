cd ./src
for layer_id in {5..5}
do
    printf "网络: ifm_operators, 层数: %d\n" $layer_id 
    python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Eyeriss --mapper MARS --type arch --version v1 --workload ifm_operators --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
done
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