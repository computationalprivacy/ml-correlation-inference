#!/bin/bash

DATASET=$1
NBR_CORES=$2

if [ $1 = "fifa19" ] || [ $1 = "fifa19_v2" ]
then 
	NBR_DATA_SAMPLES=2000
else
	NBR_DATA_SAMPLES=1000 # Default value which will be ignored for the "musk" and "communities_and_crime" datasets.
fi

echo $NBR_DATA_SAMPLES


python main.py --experiment_name=correlation_extraction  \
	--save_dir=experiments  \
	--nbr_columns=4  \
	--shadow_model_type=logreg  \
	--meta_model_type=logreg   \
	--seed=42  \
	--dataset_name=$DATASET \
       	--nbr_data_samples=$NBR_DATA_SAMPLES  \
	--nbr_repetitions=1 \
	--nbr_targets=500 \
	--nbr_cores=$NBR_CORES
