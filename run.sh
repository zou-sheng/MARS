cd ./src
# for layer_id in {0..0}
# do
#     printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..7}
# do
#     printf "网络: alexnet, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..23}
# do
#     printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..22}
# do
#     printf "网络: unet, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..16}
# do
#     printf "网络: mobilenet, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..5}
# do
#     printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..5}
# do
#     printf "网络: bertlarge, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..3}
# do
#     printf "网络: GCN_Citeseer, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

# for layer_id in {0..3}
# do
#     printf "网络: GCN_reddit, 层数: %d\n" $layer_id 
#     python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator TPU --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
# done

for layer_id in {0..7}
do
    printf "网络: alexnet, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..23}
do
    printf "网络: resnet50, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..22}
do
    printf "网络: unet, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload unet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..16}
do
    printf "网络: mobilenet, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload mobilenet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..5}
do
    printf "网络: gpt1, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..5}
do
    printf "网络: bertlarge, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload bertlarge --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..3}
do
    printf "网络: GCN_Citeseer, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload GCN_Citeseer --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done

for layer_id in {0..3}
do
    printf "网络: GCN_reddit, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload GCN_reddit --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done