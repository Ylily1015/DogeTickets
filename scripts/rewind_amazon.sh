#!/bin/bash
# 
head=(7 7 14 21 28 28 36 43 50 50 57 64 72 72 79 86 93 93 100 108 115 115 122 129)
ffn=(0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6)
for sp in `seq 0 1 23`  
do	
    python run_rewinding.py \
        --model_type cls_tuning_with_mask \
        --model_name_or_path amazon_init \
        --task_name amazon \
        --data_type combined \
        --template "" \
        --verbalizer "" \
        --data_dir datasets \
        --train_data_domains "Digital_Music" "Gift_Cards" "Movies_and_TV" "Software"\
        --test_data_domains "All_Beauty" "Automotive" "Industrial_and_Scientific" \
        --max_length 256 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --learning_rate 5e-5 \
        --weight_decay 1e-2 \
        --log_interval 1000 \
        --num_train_epochs 10 \
        --num_patience_epochs 3 \
        --warmup_proportion 0.1 \
        --max_grad_norm 1.0 \
        --lam 100 \
        --seed 776 \
        --do_rewind \
        --do_rewind_with_domain \
        --do_test \
        --head_mask_file mask-domain-100.0/mask_${head[sp]}_heads \
        --ffn_mask_file mask-domain-100.0/mask_${ffn[sp]}_ffns \
	    --sparsity head${head[sp]}_ffn${ffn[sp]}
done