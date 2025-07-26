cd ./src
# for layer_id in {74..74}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..7}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {14..14}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {20..20}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {21..21}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {75..75}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..0}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..1}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {4..4}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {15..15}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {48..48}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {20..21}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Eyeriss --mapper Cosa --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done

# for layer_id in {68..69}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {3..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {0..3}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {5..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {5..5}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload gpt3 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {8..9}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {33..33}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
# for layer_id in {20..20}
# do
#     python main.py --optim_obj latency --population 1 --epochs 1 --accelerator TensorCore --mapper Cosa --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
# done
for layer_id in {0..5}
do
    python main.py --optim_obj latency --population 1 --epochs 1 --accelerator Simba --mapper Cosa --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0
done
cd ..