#!/bin/bash


export CUDA_VISIBLE_DEVICES=$2
# dataset="and"
total_steps=20000
glove_vals=("")

# for fc_dropout in .1 .3 .5
for fc_dropout in 0 #.3
do
# for context_dropout in 0.0 .2 .3
for context_dropout in .2 #0.0
do
for init_lr in .0001 #.00001
do
for l2_weight in 0.0 #.001
# for l2_weight in 0.0
do
for decay_method in constant #cosine # linear
do
for optimizer in amsgrad #rmsprop #adam
do
for comp_dim in 64 #16 32
do
for glove in "${glove_vals[@]}"
do
for dataset in mini_norandom_0 mini_norandom_1 mini_norandom_2
do
for k in 2 3 5 10
do
    full_data="${dataset}_conj_${k}"
    case "$glove" in
    "--glove") glove_val="true" ;;
    *) glove_val="false";;
    esac
    name="fcd=${fc_dropout}_cd=${context_dropout}"`
    `"_lr=${init_lr}_wd=${l2_weight}_dm=${decay_method}_optimizer=${optimizer}"`
    `"_comp_dim=${comp_dim}_glove=${glove_val}_5conv"
    echo "Running experiment $name"
    python3 cpg_model.py -f ../shapes/Example/${full_data}/output -m $1 -d ${full_data} \
    --fc-dropout=$fc_dropout --context-dropout=$context_dropout \
    --l2-weight=$l2_weight --init-lr=$init_lr --lr-decay-method=$decay_method \
    --optimizer=$optimizer --total-steps=$total_steps \
    --s="$3/${full_data}_grid_summaries_$1/$name" --comp-hidden-dimension=$comp_dim\
    ${glove} --epochs-between-evals 15
done
done
done
done
done
done
done
done
done
done