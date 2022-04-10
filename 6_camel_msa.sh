#!/usr/bin/env bash

DESC="vanilla_msa"
MODEL_NAME="CAMeL-Lab/bert-base-arabic-camelbert-msa"
for run in {1..10};
do
	python train.py --seed $run --desc "$DESC" --model_name "$MODEL_NAME"
	# python eval.py --seed $run --desc "$DESC" --use_TAPT
	# ./eval.sh > exp_outputs/"$DESC"$run
done
