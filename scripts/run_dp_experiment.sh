#!/bin/bash

NBR_CORES=$1

for NBR_COLUMNS in 3 4 5 
do
	for EPSILON in 0\.01 0\.02 0\.05 0\.1 0\.2 0\.5 1\.0 2\.0 5\.0 10\.0 20\.0 50\.0 100\.0
	do
		python main.py --experiment_name=dp_target_attack \
			--save_dir=experiments \
			--nbr_targets=1000 \
			--nbr_columns=$NBR_COLUMNS \
			--nbr_shadow_datasets=5000 \
			--nbr_data_samples=1000 \
			--target_test_size=0.3333 \
			--nbr_data_samples_bb_aux=100 \
			--shadow_model_type=logregdp \
			--meta_model_type=logreg \
			--nbr_bins=3 \
			--seed=42 \
			--nbr_cores=$NBR_CORES \
			--constraints_scenario=column \
			--balanced_train=True \
			--balanced_test=True \
			--epsilon=$EPSILON
	done
done
