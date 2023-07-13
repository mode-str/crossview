name="convnext_tri"
data_dir="/home/crossview/university1652/train"
test_dir="/home/crossview/university1652/test"
gpu_ids="0"
lr=0.01
batchsize=8
triplet_loss=0.3
num_epochs=200
views=2

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs \

for ((j = 1; j < 3; j++));
    do
      python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
    done
