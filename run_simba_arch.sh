

cd ./src

for arch_id in {1..1}
do
    for layer_id in {0..0}  # 假设layer_id范围是0到15，根据实际情况调整
    do
        python main.py --optim_obj EDP --population 100 --epochs 100 --accelerator "Simba_${arch_id}x64" --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
        python main.py --optim_obj energy --population 100 --epochs 100 --accelerator "Simba_${arch_id}x64" --mapper MARS --type arch --version v1 --workload resnet50 --layer_id $layer_id --batch_size 1 --expanded_scope 0.1
    done
done
cd ..