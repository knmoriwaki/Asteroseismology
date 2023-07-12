#!/bin/sh

n_feature=1
seq_length=45412

model=CNN

hidden_dim=16
n_layer=2
r_drop=0
 
batch_size=128
epoch=80
epoch_decay=20
total_epoch=$(( epoch + epoch_decay ))
lr=0.002

input_id=2

loss=nllloss
output_dim=45

#loss=bce
#loss=l1norm

#model=BNN
#r_drop=0
#loss=bce

data_dir=./training_data
nrea_noise=1
ndata=1000

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
output_dir=./output/${model_base}
model_dir=${output_dir}
fname_norm=./param/norm_params.txt

mkdir -p $model_dir

### training ###
echo $model_base
today=`date '+%Y-%m-%d'`
python main.py --gpu_id 0 --isTrain --data_dir $data_dir --ndata $ndata --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $nrea_noise > ./tmp/out_${today}_${model_base}.log
echo "# output ./tmp/out_${today}_${model_base}.log"

### test ###
test_dir=./test_data
ndata_test=6
#python main.py --gpu_id 1 --test_dir $test_dir --ndata $ndata_test --model_dir_load $model_dir $model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --batch_size $batch_size --loss $loss --output_dim $output_dim --output_id 13 1

