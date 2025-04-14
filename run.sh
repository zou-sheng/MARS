cd ./src
python main.py --optim_obj latency --epochs 10 --accelerator Simba --workload alexnet --layer_id 2 --batch_size 1 --expanded_scope 0.0
cd ..