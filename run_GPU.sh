cd ./src
for layer_id in {0..7}
do
    printf "网络: alexnet, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 10 --epochs 20 --accelerator GPU --mapper MARS --type arch --version v1 --workload alexnet --layer_id $layer_id --batch_size 1 --expanded_scope 1.0 --solver lp --weight_matrix_config weight_matrix
done
