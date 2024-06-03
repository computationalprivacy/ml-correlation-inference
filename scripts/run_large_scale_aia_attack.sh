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

for ATTACK_METHOD in cia_aia_model_based cia_aia_model_less copula_base fredrikson csmia marginal_prior yeom cai wcai cia_synth_wcai_v1 cia_synth_wcai_v2 cia_synth_wcai_v3 cia_synth_wcai_v4
do
	python main.py --experiment_name=aia  \
		--nbr_columns=4  \
		--nbr_shadow_datasets=5000  \
		--nbr_data_samples_bb_aux=100  \
		--nbr_data_samples=$NBR_DATA_SAMPLES \
		--shadow_model_type=logreg  \
		--meta_model_type=logreg  \
		--nbr_bins=3  \
		--seed=42  \
		--balanced_train=True  \
		--dataset_name=$DATASET  \
		--target_test_size=0  \
		--nbr_repetitions=1  \
		--nbr_targets=1000  \
		--nbr_cores=$NBR_CORES   \
		--nbr_target_records=500   \
		--attack_method=$ATTACK_METHOD
done
