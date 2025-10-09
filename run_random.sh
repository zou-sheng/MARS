cd ./src
for layer_id in {0..0}
do
    printf "网络: resnet50, 层数: %d\n" $layer_id 
    python main.py --optim_obj EDP --population 100 --epochs 50 --accelerator Simba --mapper random --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.0 --solver lp
done
cd ..