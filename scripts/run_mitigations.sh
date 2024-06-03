#!/bin/bash

SHADOW_MODEL_TYPE=$1
META_MODEL_TYPE=$2
NBR_CORES=$3
NBR_GPUS=$4
CUDA_VISIBLE_DEVICES=$5 # Should be set to 0,1 or 0 or 1.

for NBR_COLUMNS in 3 4 5
do
	CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --experiment_name=randomized_target_attack_mitigations \
		--nbrs_significant_figures="0,1,2,3,-1" \
		--nbrs_data_samples_bb_aux=1,2,5,10,20,50,100,200,500,1000,2000,5000 \
		--save_dir=experiments \
		--nbr_targets=1000 \
		--nbr_columns=$NBR_COLUMNS \
		--nbr_shadow_datasets=5000 \
		--nbr_data_samples=1000 \
		--target_test_size=0.3333 \
		--shadow_model_type=$SHADOW_MODEL_TYPE \
		--meta_model_type=$META_MODEL_TYPE \
		--nbr_bins=3 \
		--seed=42 \
		--nbr_cores=$NBR_CORES \
		--constraints_scenario=column \
		--balanced_train=True \
		--balanced_test=True \
		--nbr_gpus=$NBR_GPUS 
done

