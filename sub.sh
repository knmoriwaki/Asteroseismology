#!/bin/sh

### data parameters ###
#data_dir=./training_data
#seq_length=45412

data_dir=./training_data_good/Spectra_ascii
comb_dir=./training_data_good
seq_length=45412

input_id=2
fname_norm=./param/norm_params_var2_ref.txt

nrea_noise=1
ndata=10000
r_train=0.9

### ML model parameters ###

model=CNN

hidden_dim=16
n_layer=12
r_drop=0.2

### training parameters ###
batch_size=128
epoch=20
epoch_decay=20
total_epoch=$(( epoch + epoch_decay ))
lr=0.001

loss=nllloss
output_dim=30

### directory names ### 

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
output_dir=./output/${model_base}
model_dir=${output_dir}

mkdir -p $model_dir


### training ###
echo $model_base
today=`date '+%Y-%m-%d'`
python main.py --gpu_id 0 --isTrain --data_dir $data_dir --comb_dir $comb_dir --ndata $ndata --r_train $r_train --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $nrea_noise --progress_bar > ./tmp/out_${today}_${model_base}.log

echo "# output ./tmp/out_${today}_${model_base}.log"

### test ###
test_dir=./Test_HBR5/Spectra_ascii
comb_dir=./Test_HBR5
ndata_test=100
python main.py --gpu_id 0 --test_dir $test_dir --comb_dir $comb_dir --ndata $ndata_test --model_dir_load $model_dir --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --batch_size $batch_size --loss $loss --output_dim $output_dim --output_id 13 5 --progress_bar

