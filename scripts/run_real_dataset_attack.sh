#!/bin/bash

DATASET=$1
SHADOW_MODEL_TYPE=$2
META_MODEL_TYPE=$3
NBR_BINS=$4
NBR_CORES=$5
NBR_GPUS=$6
CUDA_VISIBLE_DEVICES=$7 # Should be set to 0,1 if NBR_GPUS is set to 2, otherwise it should be set to either 0 or 1.


if [ $1 = "fifa19" ] || [ $1 = "fifa19_v2" ]
then 
	NBR_DATA_SAMPLES=2000
else
	NBR_DATA_SAMPLES=1000 # Default value which will be ignored for the "musk" and "communities_and_crime" datasets.
fi

echo $NBR_DATA_SAMPLES

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --experiment_name=real_dataset_attack  \
	--save_dir=experiments  \
	--nbr_columns=4  \
	--nbr_shadow_datasets=5000  \
	--nbr_data_samples_bb_aux=100  \
	--shadow_model_type=$SHADOW_MODEL_TYPE \
	--meta_model_type=$META_MODEL_TYPE   \
	--nbr_bins=$NBR_BINS   \
	--seed=42   \
	--balanced_train=True  \
	--dataset_name=$DATASET  \
	--nbr_data_samples=$NBR_DATA_SAMPLES \
	--nbr_repetitions=10  \
	--nbr_targets=100  \
	--nbr_cores=$NBR_CORES  \
	--nbr_gpus=$NBR_GPUS
