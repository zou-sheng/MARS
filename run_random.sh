cd ./src
# for layer_id in {0..75}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator Simba --mapper random --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator Simba --mapper GA --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
#     python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator Simba --mapper RL --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
# done
for layer_id in {0..75}
do
    printf "网络: deepbench, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator Eyeriss --mapper random --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator Eyeriss --mapper GA --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
    python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator Eyeriss --mapper RL --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
done
# for layer_id in {0..75}
# do
#     printf "网络: deepbench, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator TensorCore --mapper random --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TensorCore --mapper GA --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
#     python main.py --optim_obj EDP --population 100 --epochs 2000 --accelerator TensorCore --mapper RL --type arch --version v1 --workload deepbench --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
# done
cd ..