#!/bin/bash

SHADOW_MODEL_TYPE=$1
META_MODEL_TYPE=$2
NBR_CORES=$3
NBR_GPUS=$4
CUDA_VISIBLE_DEVICES=$5 # Should be set to 0,1 if NBR_GPUS is set to 2, otherwise it should be set to either 0 or 1.


for R in 0 1 2 3 4 5 6 7 8 9
do 
	for NBR_DATA_SAMPLES in 5000 4000 3000
	do 
		CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --experiment_name=real_dataset_attack  \
			--save_dir=experiments  \
			--nbr_columns=4  \
			--nbr_shadow_datasets=5000  \
			--nbr_data_samples_bb_aux=100  \
			--shadow_model_type=$SHADOW_MODEL_TYPE \
			--meta_model_type=$META_MODEL_TYPE   \
			--nbr_bins=3   \
			--seed=42   \
			--balanced_train=True  \
			--dataset_name=fifa19  \
			--nbr_data_samples=$NBR_DATA_SAMPLES \
			--nbr_targets=100  \
			--nbr_cores=$NBR_CORES  \
			--nbr_gpus=$NBR_GPUS \
			--min_repetition=$R \
			--nbr_repetitions=1
	done
done
