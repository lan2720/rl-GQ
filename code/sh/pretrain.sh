#!/usr/bin/env bash

get_name(){
    file_path=$1
    OLD_IFS="$IFS"
    IFS="/"
    arr=($file_path)
    file_name=${arr[-1]}
    IFS="."
    arr=($file_name)
    IFS="$OLD_IFS"
    dirname=${arr[0]}
    echo $dirname
}
if [ $# == 0 ] ; then 
    echo "Error: Please assign a gpu first." 
    exit 1; 
fi 
dirname="pretrain" #`get_name $0`
exp_dir="/home/jiananwang/rl-QG/exp/$dirname"

CUDA_VISIBLE_DEVICES=$1 nohup python -u run_question_generator.py \
--exp_dir=$exp_dir \
--data_path="/home/jiananwang/rl-QG/data/squad-v1/" \
--word_count_path="/home/jiananwang/rl-QG/data/squad-v1/word_counter.json" \
--glove_path="/home/jiananwang/data/glove/glove.840B.300d.txt" \
--maxium_likelihood \
--pointer_gen \
--bidir \
--save \
> ${exp_dir}/log 2>&1 &
