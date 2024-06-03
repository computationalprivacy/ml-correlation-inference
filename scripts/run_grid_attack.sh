#!/bin/bash

SHADOW_MODEL_TYPE=$1
META_MODEL_TYPE=$2
NBR_CORES=$3

python main.py --experiment_name=grid_attack \
	--save_dir=experiments \
	--nbr_columns=3 \
	--nbr_shadow_datasets=1500 \
	--nbr_data_samples=1500 \
	--shadow_test_size=0.3333 \
	--target_test_size=0 \
	--nbr_data_samples_bb_aux=100 \
	--shadow_model_type=$SHADOW_MODEL_TYPE \
	--meta_model_type=$META_MODEL_TYPE \
	--nbr_bins=3 \
	--lengths=100,100 \
	--seed=42 \
	--nbr_cores=$NBR_CORES
