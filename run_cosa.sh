cd ./src
for layer_id in {0..0}
do
    python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Simba --mapper Cosa --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
done
# for layer_id in {11..11}
# do
#     python main.py --optim_obj latency --population 100 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..22}
# do
#     python main.py --optim_obj latency --population 100 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..0}
# do
#     python main.py --optim_obj latency --population 200 --epochs 10 --accelerator Simba --mapper Cosa --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..8}
# do
#     python main.py --optim_obj latency --population 100 --epochs 50 --accelerator Simba --mapper MARS --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..0}
# do
#     python main.py --optim_obj latency --population 50 --epochs 20 --accelerator Simba --mapper Cosa --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 50 --epochs 50 --accelerator Simba_cosa --mapper Cosa --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
# done
cd ..