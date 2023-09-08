#!/bin/bash

### data parameters ###
data_dir=./training_data_good/Spectra_ascii
comb_dir=./training_data_good
seq_length=45412

input_id=2
fname_norm=./param/norm_params.txt

nrea_noise=1
ndata=256
r_train=0.99

### ML model parameters ###

model=CNN
model=Dhanpal22
model=MDN

hidden_dim=16
n_layer=6
nlayer_increase=5
r_drop=0.5

### training parameters ###
batch_size=32 #128
epoch=20
epoch_decay=20
total_epoch=$(( epoch + epoch_decay ))
lr=0.001
#lr=1e-8
l2_lambda=-1.0
#l2_lambda=0.001

loss=nllloss
output_dim=30

### directory names ### 

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
if [ `echo "$l2_lambda > 0.0" | bc` == 1 ]; then
	model_base=${model_base}_l2lambda${l2_lambda}
fi
output_dir=./output/${model_base}
model_dir=${output_dir}

mkdir -p $model_dir

gpu_id=0

### training ###
echo $model_base
today=`date '+%Y-%m-%d'`
python main.py --gpu_id $gpu_id --isTrain --data_dir $data_dir --comb_dir $comb_dir --ndata $ndata --r_train $r_train --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --nlayer_increase $nlayer_increase --r_drop $r_drop --batch_size $batch_size --l2_lambda $l2_lambda --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $nrea_noise --progress_bar # > ./tmp/out_${today}_${model_base}.log

echo "# output ./tmp/out_${today}_${model_base}.log"

### test ###
data_dir=./Test_HBR5/Spectra_ascii
comb_dir=./Test_HBR5
ndata_test=100
python main.py --gpu_id $gpu_id --data_dir $data_dir --comb_dir $comb_dir --ndata $ndata_test --model_dir_load $model_dir --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --nlayer_increase $nlayer_increase --batch_size $batch_size --loss $loss --output_dim $output_dim --output_id 13 5 --progress_bar

