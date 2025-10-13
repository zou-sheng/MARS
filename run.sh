cd ./src
for layer_id in {0..0}
do
    printf "网络: gpt1, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 20 --accelerator Simba --mapper MARS --type arch --version v1 --workload gpt1 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp --weight_matrix_config weight_matrix
done
