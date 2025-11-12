cd ./src
# python main.py --optim_obj EDP --population 100 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2
# python main.py --optim_obj EDP --population 100 --epochs 100  --accelerator Simba --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 20  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload resnet50 --layer_id 0 --batch_size 1 --expanded_scope 0.44 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload alexnet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload vgg16 --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload bert --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload unet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload retinanet --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_7B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_13B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
# python main.py --optim_obj EDP --population 20 --epochs 50  --accelerator Gemmini --mapper Saturn --type arch --version v1 --workload llama2_34B --layer_id 0 --batch_size 1 --expanded_scope 0.2 --all_DNN
cd ..