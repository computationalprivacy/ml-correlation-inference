# Source code for "Correlations inference attacks against machine learning models" 

Source code for pre-processing datasets, running experiments, and generating the figures of the "Correlations inference attacks against machine learning models" [paper](https://arxiv.org/abs/2112.08806) by Ana-Maria Cretu*, Florent Guépin*, and Yves-Alexandre de Montjoye (* denotes equal contribution). If you re-use this code please cite our paper.

## Requirements

For optimal execution we recommend using a machine with at least 40 cores and 2 GPUs. The cores are need to paralellize the execution of experiments using logistic regression models, while the GPUs are needed to parallelize the execution of experiments using multilayer perceptron models. The code can also be run using one GPU and fewer cores but it will be less efficient.

Install the environment:

```
conda env create -f correlations.yml
```

If this command returns an error, for instance due to using a different OS than Ubuntu (which we used in our experiments), you need to install the main libraries manually using `conda` or `pip`. Please refer to `correlations.yml` for the library versions. We recommend using the same version for all libraries using randomness, including `numpy`, `random`, `torch`, and `scikit-learn`.

## 1 - Figures

### Figure 1

To generate Figure 1 run the notebook ```notebooks/results/ear_shape_analysis.ipynb```.

### Figure 2

To reproduce the results of Figure 2A and 2B, run:

```
bash scripts/run_grid_attack.sh logreg logreg NBR_CORES
```

For Figure 2C, run:

```
bash scripts/run_grid_attack.sh mlptorch mlptorch NBR_CORES
```

In our experiments, we set the NBR_CORES variable to 40. You can modify it according to your resources.

To generate Figure 2, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

You can generate smaller-scale results (i.e., a blurry version of Figure 2) more quickly by reducing the granularity of the grid discretization by setting the ```--lengths``` parameter to, e.g., "10,10".

### Figure 3

Figure 3 shows results for predicting $\rho(X_1, X_2)$ from models trained on synthetic datasets of $n$ variables $X_1, ..., X_{n-1}, Y$, ($n \in [3,...,10]$).

To reproduce the results of Figure 3 - S1, run:

```
bash scripts/run_randomized_target_attack_balanced.sh logreg logreg two 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_randomized_target_attack_balanced.sh mlptorch mlptorch two 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```

Figure 3 - S2, run:

```
bash scripts/run_randomized_target_attack_balanced.sh logreg logreg column 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_randomized_target_attack_balanced.sh mlptorch mlptorch column 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```


Figure 3 - S3, run:

```
bash scripts/run_randomized_target_attack_balanced.sh logreg logreg all_but_target 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_randomized_target_attack_balanced.sh mlptorch mlptorch all_but_target 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.
For experiments using logistic regression (logreg), CUDA_VISIBLE_DEVICES should be set to 0 (default). For experiments using multilayer perceptrons (mlptorch) which are trained on the GPU, we set CUDA_VISIBLE_DEVICES to 0,1 to parallelize our code on 2 GPUs, and we set NBR_GPUS=2. If you only have 1 GPU, set CUDA_VISIBLE_DEVICES to 0 and set NBR_GPUS to 1.

To generate Figure 3, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

### Figure 4

To reproduce the results of Figure 4, run the following command:

```
bash scripts/run_mitigations.sh logreg logreg NBR_CORES 1 0
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure 4, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```

### Figure 5

To reproduce the results of Figure 5, run the following command: 

```
bash scripts/run_dp_experiment.sh NBR_CORES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure 5, run the corresponding cell in the following notebook: ```notebooks/results/dp_experiment.sh```

### Figure 6

To reproduce the results of Figure 6, run the following command:

```
bash scripts/run_correlation_extraction.sh fifa19_v2 NBR_CORES
bash scripts/run_correlation_extraction.sh communities_and_crime_v2 NBR_CORES
bash scripts/run_correlation_extraction.sh musk NBR_CORES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure 6, run the corresponding cell in the following notebook: ```notebooks/results/correlation_extraction.ipynb```

### Figure 7

The results of Figure 7 - left are already computed as a result of the experiment used to generate results for Figure - S2. To reproduce the results of Figure 7 - right, run the following command: 

```
bash scripts/run_randomized_target_attack_same_seed.sh mlptorch mlptorch column 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.
We set CUDA_VISIBLE_DEVICES to 0,1 to parallelize our code on 2 GPUs, and we set NBR_GPUS=2. If you only have 1 GPU, set CUDA_VISIBLE_DEVICES to 0 and set NBR_GPUS to 1.

To generate Figure 7, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

### Figure S1

To reproduce the results of Figure S1, run the following command:

```
bash scripts/run_randomized_target_attack_model_less_only.sh NBR_CORES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure S1, run the corresponding cell in the following notebook: ```notebooks/results/model_less_attack_analysis.ipynb```.

### Figure S2

To reproduce the results of Figure S2, run the following command:

```
bash scripts/run_randomized_target_attack_balanced.sh logreg logreg column 5 NBR_CORES 1 0
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure S2, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

### Figures S3 and S4

The results of Figures S3 and S4 are already computed as a result of the experiment used to generate results for Figure 2. To generate the figures, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

### Figure S5

To reproduce the results of Figure S5, run the following command:

```
bash scripts/run_mitigations.sh mlptorch mlptorch NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.
This experiment uses multilayer perceptrons (mlptorch) which are trained on the GPU. We set CUDA_VISIBLE_DEVICES to 0,1 to parallelize our code on 2 GPUs, and we set NBR_GPUS=2. If you only have 1 GPU, set CUDA_VISIBLE_DEVICES to 0 and set NBR_GPUS to 1.

To generate Figure S5, run the corresponding cell in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

### Figure S6

To reproduce the results of Figure S6, run the following command:

```
bash scripts/run_granularity_marginals.sh NBR_CORES
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

To generate Figure S6, run the corresponding cell in the following notebook: ```notebooks/results/real_dataset_evaluation.ipynb```

### Figures S7, S8 and S9

The results of Figures S7, S8 and S9 are already computed as a result of the experiment used to generate results for Figure 3 - S2.  To generate the figures, run the corresponding cells in the following notebook: ```notebooks/results/figures_synthetic_evaluation.ipynb```.

## 2 - Tables

The following experiments are performed on real-world datasets. Run the ```notebooks/dataset_preprocessing.ipynb``` notebook do download and pre-process the datasets.

### Table 1

To reproduce the results of Table 1, run the following command:

```
bash scripts/run_real_dataset_attack.sh communities_and_crime_v2 logreg logreg 3 NBR_CORES 1 0
bash scripts/run_real_dataset_attack.sh communities_and_crime_v2 logreg logreg 5 NBR_CORES 1 0
bash scripts/run_real_dataset_attack.sh fifa19_v2 logreg logreg 3 NBR_CORES 1 0
bash scripts/run_real_dataset_attack.sh fifa19_v2 logreg logreg 5 NBR_CORES 1 0
bash scripts/run_real_dataset_attack.sh musk logreg logreg 3 NBR_CORES 1 0
bash scripts/run_real_dataset_attack.sh musk logreg logreg 5 NBR_CORES 1 0
```

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources. 


```
bash scripts/run_real_dataset_attack.sh communities_and_crime_v2 mlptorch mlptorch 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_real_dataset_attack.sh communities_and_crime_v2 mlptorch mlptorch 5 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_real_dataset_attack.sh fifa19_v2 mlptorch mlptorch 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_real_dataset_attack.sh fifa19_v2 mlptorch mlptorch 5 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_real_dataset_attack.sh musk mlptorch mlptorch 3 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
bash scripts/run_real_dataset_attack.sh musk mlptorch mlptorch 5 NBR_CORES NBR_GPUS CUDA_VISIBLE_DEVICES
```

These commands use multilayer perceptrons (mlptorch) which are trained on the GPU. We set CUDA_VISIBLE_DEVICES to 0,1 to parallelize our code on 2 GPUs, and we set NBR_GPUS=2. If you only have 1 GPU, set CUDA_VISIBLE_DEVICES to 0 and set NBR_GPUS to 1.

The table results are aggregated in the following notebook: ```notebooks/results/real_dataset_evaluation.ipynb```. 

### Tables 2 and S1

To reproduce the results of Tables 2 and S1, run the following command:

```
bash scripts/run_large_scale_aia_attack.sh fifa19_v2 NBR_CORES
``` 

We set the NBR_CORES variable to 40 in our experiments. You can modify it according to your resources.

The table results are aggregated in the following notebook: ```notebooks/results/aia_results.ipynb```. 
