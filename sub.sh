#!/bin/sh

n_feature=1
seq_length=45412

model=CNN 
model=ResNet

hidden_dim=64
n_layer=6
r_drop=0.2
 
batch_size=32
epoch=2
epoch_decay=0
total_epoch=$(( epoch + epoch_decay ))
lr=0.001

input_id=2

loss=nllloss
output_dim=90
output_dim=45

#loss=l1norm
#loss=weighted_l1norm
#loss=bce

data_dir=./training_data_2
nrea_noise=1
ndata=2000

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
output_dir=./output/${model_base}
model_dir=${output_dir}
fname_norm=./param/norm_params.txt

mkdir -p $model_dir

### training ###
echo $model_base
today=`date '+%Y-%m-%d'`
python main.py --gpu_id 1 --isTrain --data_dir $data_dir --ndata $ndata --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 --nrea_noise $nrea_noise #> ./tmp/out_${today}_${model_base}.log
echo "# output ./tmp/out_${today}_${model_base}.log"

### transfer learning ###
input_id=1
model_dir_new=${model_dir}/tl_id${input_id}
mkdir -p $model_dir_new
#python main.py --isTrain --data_dir $data_dir --ndata $ndata --model_dir_load $model_dir --model_dir_save $model_dir_new --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 --nrea_noise $nrea_noise > ./tmp/out_tf_${today}_${model_base}.log

### test ###
test_dir=./test_data
ndata_test=6
model_file_name=model.pth
#python main.py --test_dir $test_dir --ndata $ndata_test --model_dir_load $model_dir $model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --batch_size $batch_size --loss $loss --output_dim $output_dim --output_id 13 1

