#!/bin/sh

### basic parameters (same for original and new model) ###
### also, you should set the same output_id ###
n_feature=1
seq_length=45412

model=CNN 

hidden_dim=32
n_layer=6
r_drop=0.2

fname_norm=./param/norm_params.txt

loss=nllloss
output_dim=45

### training parameters of original model ###
batch_size=128
epoch=80
epoch_decay=20
total_epoch=$(( epoch + epoch_decay ))
lr=0.001

input_id=2

data_dir=./training_data 
nrea_noise=1
ndata=8000

model_base=${model}_hd${hidden_dim}_nl${n_layer}_id${input_id}_r${r_drop}_${loss}_bs${batch_size}_ep${total_epoch}_lr${lr}_ndata${ndata}
model_dir=./output/${model_base}

### training parameters of new model ###
_batch_size=32
_epoch=10
_epoch_decay=0
_total_epoch=$(( epoch + epoch_decay ))
_lr=0.001

_input_id=1

_data_dir=./training_data_2
_nrea_noise=1
_ndata=2000

tl_name=tl_id${_input_id}

_model_dir=./output/${model_base}/${tl_name}

mkdir -p $_model_dir

### transfer learning ###
echo transfer learning
echo ${model_base}/${tl_name}
today=`date '+%Y-%m-%d'`
python main.py --gpu_id 0 --isTrain --data_dir $_data_dir --ndata $_ndata --model_dir_load $model_dir --model_dir_save $_model_dir --model ${model} --fname_norm $fname_norm --input_id $_input_id --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $_batch_size --epoch $_epoch --epoch_decay $_epoch_decay --lr $_lr --loss $loss --output_dim $output_dim --output_id 13 5 --nrea_noise $_nrea_noise --i_layer_freeze 2 3 4 5 > ./tmp/out_${tl_name}_${today}_${model_base}.log

