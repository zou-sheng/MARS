cd ./src
python main.py --optim_obj latency --epochs 10 --accelerator Simba --mapper Cosa --type arch --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2
cd ..