#!/usr/bin/env bash

DESC="embedding"
MODEL_NAME="CAMeL-Lab/bert-base-arabic-camelbert-ca"
for run in {1..10};
do
	python train.py --seed $run --desc "$DESC" --model_name "$MODEL_NAME" --embed_question --embed_ner
	# python eval.py --seed $run --desc "$DESC" --use_TAPT
	# ./eval.sh > exp_outputs/"$DESC"$run
done
