#!/usr/bin/env bash

DESC="TAPT"
MODEL_NAME="TAPT"
for run in {1..2};
do
	python train.py --seed $run --desc "$DESC" --model_name "$MODEL_NAME"
	# python eval.py --seed $run --desc "$DESC" --use_TAPT
	# ./eval.sh > exp_outputs/"$DESC"$run
done
