cd ./src
python main.py --optim_obj EDP --population 100 --epochs 20  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2
cd ..