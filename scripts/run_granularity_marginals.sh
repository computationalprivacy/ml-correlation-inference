#!/bin/bash

NBR_CORES=$1

for G in 5 10 20 50 100
do 
	python main.py --experiment_name=real_dataset_attack  \
		--save_dir=experiments  \
		--nbr_columns=4  \
		--nbr_shadow_datasets=5000  \
		--nbr_data_samples_bb_aux=100  \
		--shadow_model_type=logreg \
		--meta_model_type=logreg   \
		--nbr_bins=3   \
		--seed=42   \
		--balanced_train=True  \
		--dataset_name=fifa19_v2  \
		--nbr_data_samples=2000 \
		--nbr_repetitions=10 \
		--nbr_targets=100  \
		--nbr_cores=$NBR_CORES  \
		--nbr_marginal_bins=$G
done
