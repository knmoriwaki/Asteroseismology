#!/bin/sh

### data parameters ###

data_dir=./training_data2
seq_length=68830

input_id=1
fname_norm=./param/norm_params1.txt

nrea_noise=1
ndata=95000
r_train=0.999

### ML model parameters ###

model=CNN

hidden_dim=16
n_layer=4
r_drop=0

### training parameters ###
batch_size=128
epoch=1
epoch_decay=1
total_epoch=$(( epoch + epoch_decay ))
lr=0.002

loss=nllloss
output_dim=45

### directory names ### 

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
output_dir=./output/${model_base}
model_dir=${output_dir}

mkdir -p $model_dir

### training ###
today=`date '+%Y-%m-%d'`
nohup python main.py --gpu_id 0 --isTrain --data_dir $data_dir --ndata $ndata --r_train $r_train --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $nrea_noise > ./tmp/out_${today}_${model_base}.log 2>&1 &

