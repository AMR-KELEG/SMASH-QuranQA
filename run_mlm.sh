python run_mlm.py \
    --model_name_or_path CAMeL-Lab/bert-base-arabic-camelbert-ca \
    --train_file data/quran.txt \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --max_seq_length 768 \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir test-mlm/
