# for layer_id in {0..7}
# do
#     printf "网络: alexnet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/alexnet_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..7}
# do
#     printf "网络: alexnet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/alexnet_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..5}
# do
#     printf "网络: bertlarge, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/bertlarge_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..5}
# do
#     # printf "网络: bertlarge, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/bertlarge_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..5}
# do
#     # printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/gpt1_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..5}
# do
#     # printf "网络: gpt1, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/gpt1_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..16}
# do
#     printf "网络: mobilenet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/mobilenet_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..16}
# do
#     # printf "网络: mobilenet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/mobilenet_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..22}
# do
#     printf "网络: unet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/unet_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..22}
# do
#     # printf "网络: unet, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/unet_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..53}
# do
#     # printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/Cosa/Simba/obj_latency/resnet50_input1/layer-$layer_id/expand-0.0/timeloop-model.stats.txt --arch Simba
    
# done

# for layer_id in {0..53}
# do
#     # printf "网络: resnet50, 层数: %d\n" $layer_id 
#     python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_EDP/resnet50_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
# done

for layer_id in {0..75}
do
    # printf "网络: deepbench, 层数: %d\n" $layer_id 
    python communication_analyzer.py /home/mingchuan/Desktop/zousheng/mars/report/MARS/Simba/obj_latency/deepbench_input1/layer-$layer_id/expand-0.1/timeloop-model.stats.txt --arch Simba
    
done