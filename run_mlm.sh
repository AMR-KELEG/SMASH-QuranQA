python run_mlm.py \
    --model_name_or_path CAMeL-Lab/bert-base-arabic-camelbert-ca \
    --train_file data/quran.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --max_seq_length 768 \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir TAPT/ \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 0.000001 \
    --warmup_ratio 0.06 \
    --lr_scheduler_type "constant" \
    --logging_first_step \
    --overwrite_output_dir
