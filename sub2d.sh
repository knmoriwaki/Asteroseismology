#!/bin/sh

### data parameters ###
#seq_length=45412

data_dir=training_data_2D
seq_length=32617
seq_length_2=35

#input_id=2
#fname_norm=./param/norm_params.txt

input_id=1
fname_norm=./param/norm_params1.txt

nrea_noise=1
ndata=10000 #00
r_train=0.99

### ML model parameters ###

model=CNN

hidden_dim=16
n_layer=6 
r_drop=0

### training parameters ###
batch_size=32
epoch=1
epoch_decay=0
total_epoch=$(( epoch + epoch_decay ))
lr=0.002

loss=nllloss
output_dim=45

#loss=bce
#loss=l1norm


### directory names ### 

model_base=${model}2d_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
output_dir=./output/${model_base}
model_dir=${output_dir}

mkdir -p $model_dir


### training ###
echo $model_base
today=`date '+%Y-%m-%d'`
python main.py --gpu_id 0 --isTrain --data_dir $data_dir --ndata $ndata --comb_dir /mnt/data_cat3/yuting/Spectra/Data_16CyA_1yr_HBR5 --r_train $r_train --model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --seq_length_2 $seq_length_2 --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --epoch_decay $epoch_decay --lr $lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $nrea_noise --progress_bar #> ./tmp/out_${today}_${model_base}.log

echo "# output ./tmp/out_${today}_${model_base}.log"

### test ###
test_dir=./test_data
ndata_test=6
#python main.py --gpu_id 1 --test_dir $test_dir --ndata $ndata_test --comb_dir /mnt/data_cat3/yuting/Spectra/Data_16CyA_1yr_HBR5 --model_dir_load $model_dir $model_dir_save $model_dir --model ${model} --fname_norm $fname_norm --input_id $input_id --seq_length $seq_length --seq_length_2 $seq_length_2 --hidden_dim $hidden_dim --n_layer $n_layer --batch_size $batch_size --loss $loss --output_dim $output_dim --output_id 13 1 --progress_bar

