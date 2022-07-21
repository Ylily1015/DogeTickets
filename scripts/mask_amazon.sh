# !/bin/sh

# mask 
# use trained model, get mask file

python run_masking.py \
    --model_type cls_tuning_with_mask \
    --model_name_or_path trained_model_path \
    --task_name amazon \
    --data_type combined \
    --template "" \
    --verbalizer "" \
    --data_dir datasets \
    --train_data_domains "All_Beauty" "Industrial_and_Scientific" "Movies_and_TV" "Software" \
    --max_length 256 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --lam 100 \
    --seed 776 \
    --do_mask_with_domain \
    --normalize_by_layer \
    --percent_seq `seq 5 5 100` 
