#!/bin/bash

NBR_CORES=$1

for NBR_SAMPLES in 1 2 5 10 20 50 100 200 500 1000
do
	python main.py  --experiment_name=randomized_target_attack_model_less_only \
		--save_dir=experiments  \
		--nbr_targets=1000  \
		--nbr_columns=3  \
		--nbr_shadow_datasets=$NBR_SAMPLES \
		--nbr_bins=3 \
		--seed=42 \
		--nbr_cores=$NBR_CORES \
		--constraints_scenario=column \
		--balanced_train=True \
		--balanced_test=True
done


