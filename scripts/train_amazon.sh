# !/bin/sh


python run_training.py \
    --model_type cls_tuning_with_mask \
    --model_name_or_path amazon_init \
    --init_model_path amazon_init \
    --task_name amazon \
    --data_type combined \
    --template "" \
    --verbalizer "" \
    --data_dir datasets \
    --train_data_domains "Digital_Music" "Gift_Cards" "Movies_and_TV" "Software"\
    --test_data_domains "All_Beauty" "Automotive" "Industrial_and_Scientific"\
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
    --seed 776 \
    --do_train \
    --do_test \



    



