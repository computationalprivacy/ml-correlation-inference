#!/bin/bash

SHADOW_MODEL_TYPE=$1
META_MODEL_TYPE=$2
CONSTRAINTS_SCENARIO=$3
NBR_BINS=$4
NBR_CORES=$5
NBR_GPUS=$6
CUDA_VISIBLE_DEVICES=$7 # Should be set to 0,1 or 0 or 1.

for NBR_COLUMNS in 3 4 5
do
	CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --experiment_name=randomized_target_attack \
		--save_dir=experiments \
		--nbr_targets=1000 \
		--nbr_columns=$NBR_COLUMNS \
		--nbr_shadow_datasets=5000 \
		--nbr_data_samples=1000 \
		--target_test_size=0.3333 \
		--nbr_data_samples_bb_aux=100 \
		--shadow_model_type=$SHADOW_MODEL_TYPE \
		--meta_model_type=$META_MODEL_TYPE \
		--nbr_bins=$NBR_BINS \
		--seed=42 \
		--nbr_cores=$NBR_CORES \
		--constraints_scenario=$CONSTRAINTS_SCENARIO \
		--balanced_train=True \
		--balanced_test=True \
		--nbr_gpus=$NBR_GPUS
done

for NBR_COLUMNS in 6 7 8 9 10
do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --experiment_name=randomized_target_attack \
		--save_dir=experiments \
		--nbr_targets=1000 \
		--nbr_columns=$NBR_COLUMNS \
		--nbr_shadow_datasets=10000 \
		--nbr_data_samples=1000 \
		--target_test_size=0.3333 \
		--nbr_data_samples_bb_aux=100 \
		--shadow_model_type=$SHADOW_MODEL_TYPE \
		--meta_model_type=$META_MODEL_TYPE \
		--nbr_bins=$NBR_BINS \
		--seed=42 \
		--nbr_cores=$NBR_CORES \
		--constraints_scenario=$CONSTRAINTS_SCENARIO \
		--balanced_train=True \
		--balanced_test=True \
		--nbr_gpus=$NBR_GPUS
done

