#!/usr/bin/env bash

DESC="vanilla_less_leakage"
MODEL_NAME="CAMeL-Lab/bert-base-arabic-camelbert-ca"
for run in {1..10};
do
	python train.py --seed $run --desc "$DESC" --model_name "$MODEL_NAME"
	# zip "data/SMASH_run0$run.zip" "data/SMASH_run0$run.json"
	# ./eval.sh > exp_outputs/"$DESC"$run
done
