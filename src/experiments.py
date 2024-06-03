from collections import defaultdict
from copulas.univariate import GaussianUnivariate
import itertools
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score, mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pickle
import torch
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings("ignore")

from src.correlations_generator import generate_correlation_matrices
from src.correlations_inference import (correlation_extraction,
        correlation_inference_attack_synthetic,
        correlation_inference_attack_synthetic_model_less_only,                                
        correlation_inference_attack_synthetic_mitigations,
        correlation_inference_attack)
from src.helpers.utils import (get_device,
        get_labels,
        get_other_args,
        set_torch_seed,
        standardize_dataset)
from src.marginals import ArbitraryMarginal, compute_histogram
from src.synthetic_data import generate_dataset_from_correlation_matrix


def init_gaussian_univariates(nbr_columns):
    univariate = GaussianUnivariate()

    univariate.fit(np.random.randn(10000))
    univariate._params = {'loc': 0, 'scale': 1}
    univariates = [univariate] * nbr_columns

    return univariates


def init_univariates(dataset, K, y_binary):
    """
    Set `y_binary` to True if the last column of the dataset (by default, the
    output variable `y`) is binary.
    """
    univariates = []
    for i, column in enumerate(dataset.columns):
        binary = y_binary and (i == len(dataset.columns) - 1)
        bins, cdf = compute_histogram(dataset[column], K, binary)
        univariates.append(ArbitraryMarginal(bins, cdf, binary))
    return univariates


def correlation_inference_attack_synthetic_parallel(args):
    return correlation_inference_attack_synthetic(*args)

def correlation_inference_attack_synthetic_parallel_model_less_only(args):
    return correlation_inference_attack_synthetic_model_less_only(*args)


def correlation_inference_attack_synthetic_mitigations_parallel(args):
    return correlation_inference_attack_synthetic_mitigations(*args)


def correlation_inference_attack_parallel(args):
    return correlation_inference_attack(*args)


def correlation_extraction_parallel(args):
    return correlation_extraction(*args)


def run_attack_parallel(experiment_name, args_list, shadow_model_type,
        meta_model_type, device, nbr_cores, save_path):
    results = []

    if experiment_name in ['grid_attack', 
            'randomized_target_attack', 
            'dp_target_attack']:
        attack_parallel = correlation_inference_attack_synthetic_parallel
    elif experiment_name in ['randomized_target_attack_mitigations']:
        attack_parallel = correlation_inference_attack_synthetic_mitigations_parallel
    elif experiment_name in ['real_dataset_attack']:
        attack_parallel = correlation_inference_attack_parallel
    elif experiment_name == 'correlation_extraction':
        attack_parallel = correlation_extraction_parallel
    #addition
    elif experiment_name == 'randomized_target_attack_model_less_only':
        attack_parallel = correlation_inference_attack_synthetic_parallel_model_less_only
    else:
        raise ValueError(f'ERROR: Invalid experiment name {experiment_name}.')

    if (shadow_model_type == 'mlptorch' or meta_model_type == 'mlptorch') \
            and device == 'cuda':    
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                saved_results = pickle.load(f)
            if saved_results['done']:
                print('The experiment has already completed.')
                return saved_results
            else:
                results = saved_results['results']
                start_idx = saved_results['last_batch_end']
                print('Found checkpoint. Resuming the experiment.')
        else:
            start_idx = 0
        # Running the attack on `nbr_cores` processes at a time, to ensure that
        # the CUDA memory is freed properly. This is slower than launching
        # all processes at the same time because it waits for all `nbr_cores`
        # process to be done.
        for batch_start in range(start_idx, len(args_list), nbr_cores):
            batch_end = min(batch_start + nbr_cores, len(args_list))
            print(f'Grid cells {batch_start}-{batch_end}/{len(args_list)}')
            args_list_batch = args_list[batch_start:batch_end]
            with mp.Pool(nbr_cores) as pool:
                for result in tqdm(pool.imap(attack_parallel, args_list_batch),
                        total=len(args_list_batch)):
                    results.append(result)
            # Checkpoint the results.
            with open(save_path, 'wb') as f:
                saved_results = {'results': results, 'done': False,
                        'last_batch_end': batch_end}
                pickle.dump(saved_results, f)
    else:
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                saved_results = pickle.load(f)
            return saved_results
        with mp.Pool(nbr_cores) as pool:
            for result in tqdm(pool.imap(attack_parallel, args_list), total=len(args_list)):
                results.append(result)
    return results


def get_prefix(args):
    prefix = f'ns-{args.nbr_shadow_datasets}' +\
            f'_nds-{args.nbr_data_samples}' +\
            f'_tts-{args.target_test_size}' +\
            f'_sts-{args.shadow_test_size}' +\
            f'_ndsbb-{args.nbr_data_samples_bb_aux}' +\
            f'_smt-{args.shadow_model_type}'
    if args.shadow_model_type == 'logregdp':
        prefix += f'_eps-{args.epsilon}'
    prefix += f'_mmt-{args.meta_model_type}' +\
            f'_nb-{args.nbr_bins}' +\
            f'_sgf-{args.nbr_significant_figures}'
    return prefix


def get_prefix_mitigations(args):
    prefix = f'ns-{args.nbr_shadow_datasets}' +\
            f'_nds-{args.nbr_data_samples}' +\
            f'_tts-{args.target_test_size}' +\
            f'_sts-{args.shadow_test_size}' +\
            f'_ndsbb-{args.nbrs_data_samples_bb_aux}' +\
            f'_smt-{args.shadow_model_type}'
    if args.shadow_model_type == 'logregdp':
        prefix += f'_eps-{args.epsilon}'
    prefix += f'_mmt-{args.meta_model_type}' +\
            f'_nb-{args.nbr_bins}' +\
            f'_sgf-{args.nbrs_significant_figures}'
    return prefix



def get_prefix_correlation_extraction(args):
    prefix = f'smt-{args.shadow_model_type}'+\
            f'_nds-{args.nbr_data_samples}'
    return prefix


def get_prefix_model_less_only(args):
    prefix = f'nb-{args.nbr_bins}' +\
            f'_ns-{args.nbr_shadow_datasets}'
    return prefix


def experiment_grid_attack(save_dir, args):
    """
    Experiment where the grid in `nbr_columns`-1 dimensions is discretized 
    by dividing the [-1, 1] interval in 2*`lengths[i]` intervals. This allows
    to consider all the possible values for ( Corr(X1, Y), ..., Corr(Xn-1, Y) ).

    For each cell, we sample correlation matrices with constraints Corr(Xi, Y) 
    falling in the cell. We train and evaluate a meta model using 5-fold cross 
    validation.

    The experiment is parallelized over the various cells.
    """
    np.random.seed(args.seed)
    set_torch_seed(args.shadow_model_type, args.device, args.seed)

    univariates = init_gaussian_univariates(args.nbr_columns)

    assert len(args.lengths) == args.nbr_columns - 1
    # The list of bounds along each dimension. Each element contains the 
    # lower bounds for a cell in the `nbr_columns`-1 dimensional space.
    bounds_list = [[i/l for i in range(-l, l)] for l in args.lengths]
    #print('bounds_list', bounds_list)

    use_kfold = True
    args_list = []

    # Enumerating all the cells.
    for i, bounds in enumerate(itertools.product(*bounds_list)):
        seed_b = np.random.randint(10**8)
        #if bounds != (99/100, 33/100):
        #    continue
        device = get_device(args.device, args.nbr_gpus, i)
        args_list.append((
            univariates,
            args.nbr_columns,
            args.nbr_shadow_datasets,
            args.nbr_data_samples,
            args.shadow_test_size,
            args.nbr_data_samples_bb_aux,
            args.shadow_model_type,
            device,
            args.meta_model_type,
            args.nbr_bins,
            bounds,
            args.lengths,
            'column', # Default constraints scenario.
            False, # No need to balance the train data; it is already balanced.
            seed_b, # Seed.
            use_kfold,
            None,
            args.nbr_significant_figures,
            args.verbose, # Verbose., 
            args.epsilon,
            args.same_seed
            ))

    save_path = f'{save_dir}/l-{args.lengths}_{get_prefix(args)}.pickle'
    results = run_attack_parallel(args.experiment_name, args_list, \
            args.shadow_model_type, args.meta_model_type, args.device, \
            args.nbr_cores, save_path)

    # The results have not yet been aggregated.
    if isinstance(results, list):
        attack_results = {'args': args,
            'bounds_list': bounds_list,
            'results': dict(),
            'accs_shadow_train': dict(),
            'accs_shadow_test': dict()}
        for result in results:
            bounds, largest_bin_results, uniform_empirical_prior_results, \
                access_to_model_results, accs_models = result
            attack_results['results'][bounds] = {
                'largest_bin_results': largest_bin_results,
                'uniform_prior_empirical_results': \
                        uniform_empirical_prior_results}
            attack_results['results'][bounds].update({
                fname: access_to_model_results[fname] \
                    for fname in access_to_model_results})
            attack_results['accs_shadow_train'][bounds] = np.mean(accs_models[0])
            attack_results['accs_shadow_test'][bounds] = np.mean(accs_models[1])
    else:

        # The results have already been aggregated, just print them.
        attack_results = results

    #print(attack_results['results'])
    print(f'\nAccuracy of shadow models:')
    mean_acc_train = np.mean(list(attack_results['accs_shadow_train'].values()))
    std_acc_train = np.std(list(attack_results['accs_shadow_train'].values()))
    mean_acc_test = np.mean(list(attack_results['accs_shadow_test'].values()))
    std_acc_test = np.std(list(attack_results['accs_shadow_test'].values()))
    print(f'train={mean_acc_train:.1%}+-{std_acc_train:.1%}',
            f'test={mean_acc_test:.1%}+-{std_acc_test:.1%}\n')

    print('Attack acc:')
    # Printing the aggregated results.
    for fname in ['largest_bin_results', 'uniform_prior_empirical_results']:
        acc = np.mean([attack_results['results'][b][fname]['acc']
            for b in attack_results['results']])
        print(f'{fname}: {acc:.2%}')
    for fname in ['model_predictions', 'model_weights',
            'model_weights_canonical', 'combined']:
        try:
            test_accs = [attack_results['results'][b][fname][0]
                for b in attack_results['results']]
            train_accs = [attack_results['results'][b][fname][2]
                for b in attack_results['results']]
            print(f'{fname}: test acc={np.mean(test_accs):.2%},',
                f'train acc={np.mean(train_accs):.2%}')
        except:
            continue
    save_path = f'{save_dir}/l-{args.lengths}_{get_prefix(args)}.pickle'
    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        attack_results['done'] = True
        pickle.dump(attack_results, f)


def get_bounds(correlation_matrix, constraints_scenario):
    # Extract the correlation constraints. Depending on the attack assumptions,
    # the bounds are equal to:.
    # 1. the last column (excluding the last value, corresponding to 
    # rho(X1, Y), ..., rho(Xn-1, Y)).
    # 2. the first two values of the last column (corresponding to rho(X1, X2)).
    # 3. the entire correlation matrix, excluding rho(X1, X2).
    if constraints_scenario == 'column':
        bounds = correlation_matrix[:-1, -1]
    elif constraints_scenario == 'two':
        bounds = correlation_matrix[:2, -1]
    elif constraints_scenario == 'all_but_target':
        bounds = correlation_matrix
    else:
        raise ValueError(f'ERROR: Invalid constrints scenario {constraints_scenario}.')
    return bounds


def experiment_randomized_target_attack(save_dir, args):
    save_dir = os.path.join(save_dir, args.constraints_scenario)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/seed_{args.same_seed}_nt-{args.nbr_targets}_{get_prefix(args)}.pickle'

    np.random.seed(args.seed)
    set_torch_seed(args.shadow_model_type, args.device, args.seed)

    univariates = init_gaussian_univariates(args.nbr_columns)

    # Sample `nbr_targets` correlation matrices.
    target_correlation_matrices = generate_correlation_matrices(
            args.nbr_columns, args.nbr_targets, bounds=None, lengths=None,
            balanced=args.balanced_test)
    #print('Target correlation matrices (initial)', target_correlation_matrices)

    target_labels = get_labels(target_correlation_matrices, 0, 1,
            args.nbr_bins, args.meta_model_type)

    target_correlations = [target_correlation_matrix[0,1]
            for target_correlation_matrix in target_correlation_matrices]
    count_per_class = [np.sum(target_labels==b) for b in range(args.nbr_bins)]
    print('Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(args.nbr_bins)]))
    if args.verbose:
        print(target_labels)
    #print(target_correlation_matrices)

    target_nbr_data_samples = int( args.nbr_data_samples/ (1-args.target_test_size) + 1)
    print('Target nbr data samples', target_nbr_data_samples)
    target_datasets = [generate_dataset_from_correlation_matrix(
            correlation_matrix, univariates, target_nbr_data_samples, 
            args.target_test_size) 
            for correlation_matrix in target_correlation_matrices]
    print('Target train and test sizes:', len(target_datasets[0][0]), len(target_datasets[0][2]))

    args_list = []
    # Enumerating all the cells.
    for i, target_correlation_matrix in enumerate(target_correlation_matrices):
        seed_b = np.random.randint(10**8)
        bounds = get_bounds(target_correlation_matrix,
                args.constraints_scenario)
        if args.verbose:
            print(i, 'target', target_correlation_matrix[0,1],
                    'Bounds', bounds, 'seed', seed_b)
        device = get_device(args.device, args.nbr_gpus, i) 
        #print('bounds', bounds)
        args_list.append((univariates,
            args.nbr_columns,
            args.nbr_shadow_datasets,
            args.nbr_data_samples,
            args.shadow_test_size,
            args.nbr_data_samples_bb_aux,
            args.shadow_model_type,
            device,
            args.meta_model_type,
            args.nbr_bins,
            bounds,
            None, # lengths
            args.constraints_scenario,
            args.balanced_train, # Whether to balance the shadow_matrices.
            seed_b, # seed
            False, # use_kfold
            target_datasets[i],
            args.nbr_significant_figures,
            args.verbose, # verbose
            args.epsilon,
            args.same_seed))

    print(args.same_seed, 'Same seed', type(args.same_seed))
    results = run_attack_parallel(args.experiment_name, args_list,
            args.shadow_model_type, args.meta_model_type, args.device,
            args.nbr_cores, save_path)

    methods = ['largest_bin', 'uniform_prior_empirical', 'model_weights']
    if args.shadow_model_type == 'mlptorch':
        methods += ['model_weights_canonical']
    methods += ['model_predictions', 'combined']
    if args.shadow_model_type == 'decisiontree':
        methods = ['largest_bin', 'uniform_prior_empirical',
                'model_predictions']

    if isinstance(results, list):
        # Aggregate the results.
        attack_results = {'args': args, 'bounds': [],
                'pred_labels': {method: [] for method in methods},
                'meta_model_acc': {key: {method: [] for method in methods}
                    for key in ['train', 'test']},
                'target_correlations': target_correlations,
                'accs_shadow_train': [],
                'accs_shadow_test': [],
                'accs_target_train': [],
                'accs_target_test': []
                }
        for i, result in enumerate(results):
            bounds, largest_bin_results, uniform_prior_empirical_results, \
                    access_to_ml_results, accs_models = result
            attack_results['bounds'].append(bounds)
            # Results for the no ML attacks.
            attack_results['pred_labels']['largest_bin'].append(
                largest_bin_results['pred'])
            attack_results['pred_labels']['uniform_prior_empirical'].\
                append(uniform_prior_empirical_results['pred'])
            for fname in access_to_ml_results:
                pred_label, accs = access_to_ml_results[fname]
                attack_results['pred_labels'][fname].append(pred_label)
                attack_results['meta_model_acc']['test'][fname].append(accs[0])
                attack_results['meta_model_acc']['train'][fname].append(accs[2])

            attack_results['accs_shadow_train'].append(np.mean(accs_models[0]))
            attack_results['accs_shadow_test'].append(np.mean(accs_models[1]))
            attack_results['accs_target_train'].append(accs_models[2])
            attack_results['accs_target_test'].append(accs_models[3])
        attack_results['target_labels'] = target_labels
        #print('target labels', target_labels)

        attack_results['accuracy'] = dict()
        for method in methods:
            pred_labels = attack_results['pred_labels'][method]
            #print(method, 'predicted labels', pred_labels)
            attack_results['accuracy'][method] = \
                accuracy_score(target_labels, pred_labels)
    else:
        # The results have already been aggregated.
        attack_results = results

    for key in ['shadow', 'target']:
        print(f'\nAccuracy of {key} models:')
        mean_acc_train = np.mean(attack_results[f'accs_{key}_train'])
        std_acc_train = np.std(attack_results[f'accs_{key}_train'])
        mean_acc_test = np.mean(attack_results[f'accs_{key}_test'])
        std_acc_test = np.std(attack_results[f'accs_{key}_test'])
        print(f'train={mean_acc_train:.1%}+-{std_acc_train:.1%}',
                f'test={mean_acc_test:.1%}+-{std_acc_test:.1%}') 

    print(f'\nMeta model acc (avg over {args.nbr_targets} targets):')
    for fname in methods[2:]:
        mean_test_acc = np.mean(attack_results['meta_model_acc']['test'][fname])
        std_test_acc = np.std(attack_results['meta_model_acc']['test'][fname])
        mean_train_acc = np.mean(
                attack_results['meta_model_acc']['train'][fname])
        std_train_acc = np.std(attack_results['meta_model_acc']['train'][fname])
        print(f'   {fname}: test={mean_test_acc:.1%}+-{std_test_acc:.1%}, train={mean_train_acc:.1%}+-{std_train_acc:.1%}.')
    
    print('\nAttack acc: ')
    for method, acc in attack_results['accuracy'].items():
        print(method, f'{acc:.1%}')

    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        attack_results['done'] = True
        pickle.dump(attack_results, f)


def experiment_randomized_target_attack_mitigations(save_dir, args):
    """
    Same as above, except that we run all the mitigations in the same
    experiment (i.e., only train the shadow models once before applying 
    mitigations to the features). This code only runs the black-box attack 
    (which the mitigations aim to prevent).
    """

    # Mitigation #1: Return confidence scores with fewer significant figures.
    nbrs_significant_figures = [int(n) 
            for n in args.nbrs_significant_figures.split(',')]
    # Mitigation #2: Restrict the number of times the model can be queried.
    nbrs_data_samples_bb_aux = [int(n)
            for n in args.nbrs_data_samples_bb_aux.split(',')]

    save_dir = os.path.join(save_dir, args.constraints_scenario)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/nt-{args.nbr_targets}_{get_prefix_mitigations(args)}.pickle'

    np.random.seed(args.seed)
    set_torch_seed(args.shadow_model_type, args.device, args.seed)

    univariates = init_gaussian_univariates(args.nbr_columns)

    # Sample `nbr_targets` correlation matrices.
    target_correlation_matrices = generate_correlation_matrices(
            args.nbr_columns, args.nbr_targets, bounds=None, lengths=None,
            balanced=args.balanced_test)
    #print('Target correlation matrices (initial)', target_correlation_matrices)

    target_labels = get_labels(target_correlation_matrices, 0, 1,
            args.nbr_bins, args.meta_model_type)

    target_correlations = [target_correlation_matrix[0,1]
            for target_correlation_matrix in target_correlation_matrices]
    count_per_class = [np.sum(target_labels==b) for b in range(args.nbr_bins)]
    print('Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(args.nbr_bins)]))
    if args.verbose:
        print(target_labels)
    #print(target_correlation_matrices)

    target_nbr_data_samples = int( args.nbr_data_samples/ (1-args.target_test_size) + 1)
    print('Target nbr data samples', target_nbr_data_samples)
    target_datasets = [generate_dataset_from_correlation_matrix(
        correlation_matrix, univariates, target_nbr_data_samples,
        args.target_test_size)
        for correlation_matrix in target_correlation_matrices]
    print('Target train and test sizes:', len(target_datasets[0][0]), len(target_datasets[0][2]))

    args_list = []
    # Enumerating all the cells.
    for i, target_correlation_matrix in enumerate(target_correlation_matrices):
        seed_b = np.random.randint(10**8)
        bounds = get_bounds(target_correlation_matrix,
                args.constraints_scenario)
        if args.verbose:
            print(i, 'target', target_correlation_matrix[0,1],
                    'Bounds', bounds, 'seed', seed_b)
        device = get_device(args.device, args.nbr_gpus, i)
        #print('bounds', bounds)
        args_list.append((univariates,
            args.nbr_columns,
            args.nbr_shadow_datasets,
            args.nbr_data_samples,
            nbrs_data_samples_bb_aux, # Vary the number of queries.
            args.shadow_model_type,
            device,
            args.meta_model_type,
            args.nbr_bins,
            bounds,
            args.constraints_scenario,
            args.balanced_train, # Whether to balance the shadow matrices.
            seed_b, # seed
            target_datasets[i],
            nbrs_significant_figures, # Vary the number of significant figures.
            args.verbose, # verbose
            ))

    results = run_attack_parallel(args.experiment_name, args_list,
            args.shadow_model_type, args.meta_model_type, args.device,
            args.nbr_cores, save_path)

    mitigations = ['nbr_significant_figures', 'nbr_queries']
    nbrs_mitigations = [len(nbrs_significant_figures), 
            len(nbrs_data_samples_bb_aux)]

    if isinstance(results, list):
        # Aggregate the results.
        attack_results = {'args': args, 'bounds': [],
                'pred_labels': {m: np.zeros((args.nbr_targets, n))
                    for m, n in zip(mitigations, nbrs_mitigations)},
                'target_correlations': target_correlations
                }

        for i, result in enumerate(results):
            bounds, access_to_ml_results = result
            attack_results['bounds'].append(bounds)
            
            # Collect the results when the number of significant figures,
            # respectively the number of queries is restricted.
            for j, m in enumerate(mitigations):
                attack_results['pred_labels'][m][i] = access_to_ml_results[j]

        attack_results['target_labels'] = target_labels
        #print('target labels', target_labels)

        attack_results['accuracy'] = {m: dict() for m in mitigations}
        for j, m in enumerate(mitigations):
            for n in range(nbrs_mitigations[j]):
                pred_labels = attack_results['pred_labels'][m][:, n]
                if m == 'nbr_significant_figures':
                    key = nbrs_significant_figures[n]
                elif m == 'nbr_queries':
                    key = nbrs_data_samples_bb_aux[n]
                else:
                    raise ValueError(f'Unknown mitigation {m}')
                attack_results['accuracy'][m][key] = \
                    accuracy_score(target_labels, pred_labels)
    else:
        # The results have already been aggregated.
        attack_results = results

    print('Attack acc for varying number of significant figures (100 queries):')
    for nsf, acc in attack_results['accuracy']['nbr_significant_figures'].\
            items():
        print(nsf, f'{acc:.1%}')
    print('Attack acc for varying number of queries and nsf=-1 (all): ')
    for nq, acc in attack_results['accuracy']['nbr_queries'].items():
        print(nq, f'{acc:.1%}')

    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        attack_results['done'] = True
        pickle.dump(attack_results, f)


def generate_triplets(n):
    """
    Generates all the unique triplets consisting of a pair of variables (X1, X2)
    and a third variable X3 that is different from the first two.

    The order of the variables in the pair does not matter, so we stick to the
    default ordering i<j.
    """
    assert n >= 3, f'ERROR: Cannot select triplets for {n} columns.'
    triplets = []
    for i in range(n-1):
        for j in range(i+1, n):
            for k in range(n):
                if k != i and k != j:
                    triplets.append( (i, j, k) )
    return triplets


def experiment_real_dataset(save_dir, args, seed_repetition):
    assert args.constraints_scenario == 'column', \
            'ERROR: Invalid {args.constraints_scenario}.'

    save_dir = os.path.join(save_dir, f'nmb-{args.nbr_marginal_bins}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/sr-{seed_repetition}_nt-{args.nbr_targets}_{get_prefix(args)}.pickle'

    np.random.seed(seed_repetition)

    # Load the dataset.
    dataset_path = os.path.join(f'{args.datasets_dir}/{args.dataset_name}/dataset.csv')
    dataset = pd.read_csv(dataset_path)
    assert 'y' in dataset.columns, 'ERROR: The dataset has no `y` variable.'
    total_nbr_columns = len(dataset.columns)
    print(f'Successfully loaded the {args.dataset_name} dataset containing',
            f'{total_nbr_columns} columns and {len(dataset)} samples.\n')

    # We run the attack against datasets of 4 columns (including the output
    # variable `y`). We generate all the possible triplets consisting of a
    # pair of variables (X1 and X2, whose correlation we aim to infer) and a
    # third variable.
    assert args.nbr_columns == 4, 'ERROR: The number of columns used in the attack against the real datasets should be 4.'
    all_triplets = generate_triplets(total_nbr_columns-1)
    print(f'The dataset yields a total of {len(all_triplets)} unique target',
            'correlations (pairs of variables) coupled with a third variable.')
    # We shuffle the triplets.
    all_triplets = np.random.permutation(all_triplets)
    #print(all_triplets[0])

    # When the output variable of the dataset is not binary (but continuous),
    # we only map it to a binary variable for the purpose of training the 
    # classification model.
    #y_binary = True if args.dataset_name == 'musk' else False
    #print(f'In the dataset {args.dataset_name}, y_binary={y_binary}.')
    y_binary = True

    args_list = []
    target_correlation_matrices = []
    for ti in range(args.nbr_targets):
        seed_b = np.random.randint(10**8)
        triplet = all_triplets[ti]
        X_cols = [dataset.columns[t] for t in triplet]
        target_dataset = dataset[X_cols + ['y']]
        if args.dataset_name in ['fifa19', 'fifa19_v2']:
            # We downsample records from the FIFA19 dataset.
            target_dataset = target_dataset.sample(args.nbr_data_samples,
                    random_state=seed_b)
        target_dataset = standardize_dataset(target_dataset, y_binary)
        nbr_data_samples = len(target_dataset)
        #print(len(target_dataset), target_dataset.head())
        univariates = init_univariates(target_dataset,
                args.nbr_marginal_bins, y_binary)

        # The format passed to the correlation inference attack.
        target_dataset_arg = (
                target_dataset[X_cols], # X_train
                target_dataset['y'], # y_train
                [], #X_test
                []  #y_test
                )

        target_correlation_matrix = target_dataset.corr().to_numpy()
        target_correlation_matrices.append(target_correlation_matrix)

        bounds = get_bounds(target_correlation_matrix,
                args.constraints_scenario)
        #print(bounds)

        device = get_device(args.device, args.nbr_gpus, ti) 
        #print('bounds', bounds)
        args_list.append((
            target_correlation_matrix[0,1],
            univariates,
            args.nbr_columns,
            args.nbr_shadow_datasets,
            nbr_data_samples,
            args.nbr_data_samples_bb_aux,
            args.shadow_model_type,
            device,
            args.meta_model_type,
            args.nbr_bins,
            bounds,
            args.constraints_scenario,
            args.balanced_train, # Whether to balance the shadow matrices.
            seed_b, # seed
            False, # use_kfold
            target_dataset_arg,
            args.nbr_significant_figures,
            args.verbose
            ))


    target_labels = get_labels(target_correlation_matrices, 0, 1,
            args.nbr_bins, args.meta_model_type)

    count_per_class = [np.sum(target_labels==b) for b in range(args.nbr_bins)]
    print('Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(args.nbr_bins)]))
    if args.verbose:
        print(target_labels)

    results = run_attack_parallel(args.experiment_name,
            args_list,
            args.shadow_model_type,
            args.meta_model_type,
            args.device,
            args.nbr_cores,
            save_path)

    methods = ['largest_bin', 'uniform_prior_empirical', 'model_predictions']
    if args.shadow_model_type != 'decisiontree':
        methods.append('model_weights')
        if args.shadow_model_type == 'mlptorch' \
                or args.shadow_model_type == 'mlp':
            methods.append('model_weights_canonical')
        methods.append('combined')

    if isinstance(results, list):
        attack_results = {'args': args,
            'bounds': [],
            'shadow_bounds': [],
            'columns': all_triplets[:args.nbr_targets],
            'pred_labels': {method: [] for method in methods},
            'target_correlations': [tcm[0,1]
                for tcm in target_correlation_matrices],
            'target_labels': target_labels,
            'meta_model_acc': {key: {method: [] for method in methods}
                for key in ['train', 'test']}}

        for i, result in enumerate(results):
            (bounds, shadow_bounds), largest_bin_results, \
                    uniform_prior_empirical_results, access_to_ml_results \
                    = result[0], result[1], result[2], result[3]
            attack_results['bounds'].append(bounds)
            attack_results['shadow_bounds'].append(shadow_bounds)

            # Results for the no ML attacks.
            attack_results['pred_labels']['largest_bin'].append(
                    largest_bin_results['pred'])
            attack_results['pred_labels']['uniform_prior_empirical'].\
                    append(uniform_prior_empirical_results['pred'])
            for fname in access_to_ml_results:
                pred_label, accs = access_to_ml_results[fname]
                attack_results['pred_labels'][fname].append(pred_label)
                attack_results['meta_model_acc']['test'][fname].append(accs[0])
                attack_results['meta_model_acc']['train'][fname].append(accs[2])
    else:
        # The results have already been aggregated.
        attack_results = results

    print(f'Meta model acc (avg over {args.nbr_targets} targets:')
    for fname in methods[2:]:
        mean_test_acc = np.mean(attack_results['meta_model_acc']['test'][fname])
        std_test_acc = np.std(attack_results['meta_model_acc']['test'][fname])
        mean_train_acc = np.mean(
                attack_results['meta_model_acc']['train'][fname])
        std_train_acc = np.std(attack_results['meta_model_acc']['train'][fname])
        print(f'   {fname}: test={mean_test_acc:.1%}+-{std_test_acc:.1%}, train={mean_train_acc:.1%}+-{std_train_acc:.1%}.')

    attack_results['accuracy'] = dict()
    for method in methods:
        pred_labels = attack_results['pred_labels'][method]
        attack_results['accuracy'][method] = \
                accuracy_score(target_labels, pred_labels)

    print('Attack accuracy: ')
    for method, acc in attack_results['accuracy'].items():
        print(method, f'{acc:.2%}')

    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        attack_results['done'] = True
        pickle.dump(attack_results, f)


def experiment_correlation_extraction(save_dir, args):
    save_dir = os.path.join(save_dir, f'nmb-{args.nbr_marginal_bins}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/nt-{args.nbr_targets}_{get_prefix_correlation_extraction(args)}.pickle'

    np.random.seed(args.seed)

    # Load the dataset.
    dataset_path = os.path.join(f'{args.datasets_dir}/{args.dataset_name}/dataset.csv')
    dataset = pd.read_csv(dataset_path)
    assert 'y' in dataset.columns, 'ERROR: The dataset has no `y` variable.'
    total_nbr_columns = len(dataset.columns)
    print(f'Successfully loaded the {args.dataset_name} dataset containing',
            f'{total_nbr_columns} columns and {len(dataset)} samples.\n')

    # We run the attack against datasets of 4 columns (including the output
    # variable `y`). We generate all the possible triplets consisting of a
    # pair of variables (X1 and X2, whose correlation we aim to infer) and a
    # third variable.
    assert args.nbr_columns == 4, 'ERROR: The number of columns used in the attack against the real datasets should be 4.'
    all_triplets = generate_triplets(total_nbr_columns-1)
    print(f'The dataset yields a total of {len(all_triplets)} unique target',
            'correlations (pairs of variables) coupled with a third variable.')
    # We shuffle the triplets.
    all_triplets = np.random.permutation(all_triplets)
    #print(all_triplets[0])

    # When the output variable of the dataset is not binary (but continuous),
    # we only map it to a binary variable for the purpose of training the 
    # classification model.
    #y_binary = True if args.dataset_name == 'musk' else False
    #print(f'In the dataset {args.dataset_name}, y_binary={y_binary}.')
    y_binary = True

    args_list = []
    rhos_true = []
    nbrs_queries = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for ti in range(args.nbr_targets):
        seed_b = np.random.randint(10**8)
        triplet = all_triplets[ti]
        X_cols = [dataset.columns[t] for t in triplet]
        target_dataset = dataset[X_cols + ['y']]
        if args.dataset_name in ['fifa19', 'fifa19_v2']:
            # We downsample records from the FIFA19 dataset.
            target_dataset = target_dataset.sample(args.nbr_data_samples,
                    random_state=seed_b)
        target_dataset = standardize_dataset(target_dataset, y_binary)
        nbr_data_samples = len(target_dataset)
        #print(len(target_dataset), target_dataset.head())
        univariates = init_univariates(target_dataset,
                args.nbr_marginal_bins, y_binary)

        # The format passed to the correlation inference attack.
        target_dataset_arg = (
                target_dataset[X_cols], # X_train
                target_dataset['y'], # y_train
                [], #X_test
                []  #y_test
                )

        target_correlation_matrix = target_dataset.corr().to_numpy()
        rhos_true.append(target_correlation_matrix[-1, :-1])

        assert args.shadow_model_type == 'logreg', f'ERROR: Invalid --shadow_model_type={args.shadow_model_type}.'
        device = 'cpu'

        args_list.append( (univariates, 
            target_dataset_arg, 
            args.shadow_model_type,
            nbrs_queries,
            seed_b,
            device))

    results = run_attack_parallel(args.experiment_name,
            args_list,
            args.shadow_model_type,
            args.meta_model_type,
            args.device,
            args.nbr_cores,
            save_path)

    methods = ['ours', 'random_guess']

    extraction_results = {'args': args,
            'nbr_queries': nbrs_queries,
            'rhos_true': rhos_true,
            'rhos_pred': {method: {nbr_queries: [] 
                for nbr_queries in nbrs_queries} for method in methods},
            'mses': {method: {nbr_queries: [] 
                for nbr_queries in nbrs_queries} for method in methods},
            'mean_mse': {method: defaultdict(float) for method in methods},
            'ci_lower_mse': {method: defaultdict(float) for method in methods},
            'ci_upper_mse': {method: defaultdict(float) for method in methods}
            }

    if isinstance(results, list):
        for i, result in enumerate(results):
            for method in methods:
                for qi, nbr_queries in enumerate(nbrs_queries):
                    #print(result, method, qi)
                    rho_pred = result[method][qi]
                    #print(i, qi, method, rho_pred, rhos_true[i])
                    mse = MSE(rho_pred, rhos_true[i])
                    extraction_results['rhos_pred'][method][nbr_queries].\
                        append(rho_pred)
                    extraction_results['mses'][method][nbr_queries].append(mse)
    #print(extraction_results)
    else:
        extraction_results = results
     
    for method in methods:
        print(f'Method={method}, avg MSE over {args.nbr_targets} target models')
        for nbr_queries in nbrs_queries:
            mses = extraction_results['mses'][method][nbr_queries]
            #print(mses)
            mean_mse = np.mean(mses)
            ci_lower_mse, ci_upper_mse = st.t.interval(alpha=0.95, 
                    df=len(mses)-1, loc=np.mean(mses), scale=st.sem(mses))
            error = mean_mse - ci_lower_mse
            print(f'{nbr_queries} queries: {mean_mse:.3f}+-{error:.3f}')
            extraction_results['mean_mse'][method][nbr_queries] = mean_mse
            extraction_results['ci_lower_mse'][method][nbr_queries] = \
                    ci_lower_mse
            extraction_results['ci_upper_mse'][method][nbr_queries] = \
                    ci_upper_mse
        

    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        extraction_results['done'] = True
        pickle.dump(extraction_results, f)

 

def experiment_randomized_target_attack_model_less_only(save_dir, args):
    save_dir = os.path.join(save_dir, args.constraints_scenario)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/nt-{args.nbr_targets}_{get_prefix_model_less_only(args)}.pickle'

    np.random.seed(args.seed)
    set_torch_seed(args.shadow_model_type, args.device, args.seed)

    univariates = init_gaussian_univariates(args.nbr_columns)

    # Sample `nbr_targets` correlation matrices.
    target_correlation_matrices = generate_correlation_matrices(
            args.nbr_columns, args.nbr_targets, bounds=None, lengths=None,
            balanced=args.balanced_test)
    #print('Target correlation matrices (initial)', target_correlation_matrices)

    target_labels = get_labels(target_correlation_matrices, 0, 1,
            args.nbr_bins, args.meta_model_type)

    target_correlations = [target_correlation_matrix[0,1]
            for target_correlation_matrix in target_correlation_matrices]
    count_per_class = [np.sum(target_labels==b) for b in range(args.nbr_bins)]
    print('Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(args.nbr_bins)]))
    if args.verbose:
        print(target_labels)
    #print(target_correlation_matrices)

    args_list = []
    # Enumerating all the cells.
    for i, target_correlation_matrix in enumerate(target_correlation_matrices):
        seed_b = np.random.randint(10**8)
        bounds = get_bounds(target_correlation_matrix,
                args.constraints_scenario)
        if args.verbose:
            print(i, 'target', target_correlation_matrix[0,1],
                    'Bounds', bounds, 'seed', seed_b)
        device = get_device(args.device, args.nbr_gpus, i) 
        #print('bounds', bounds)
        args_list.append((
            args.nbr_columns,
            args.nbr_shadow_datasets,
            args.meta_model_type,
            args.nbr_bins,
            bounds,
            args.constraints_scenario,
            args.balanced_train, # Whether to balance the shadow_matrices.
            seed_b, # seed
            args.verbose # verbose
            ))

    results = run_attack_parallel(args.experiment_name, args_list,
            args.shadow_model_type, args.meta_model_type, args.device,
            args.nbr_cores, save_path)

    methods = ['largest_bin', 'uniform_prior_empirical']

    if isinstance(results, list):
        # Aggregate the results.
        attack_results = {'args': args, 'bounds': [],
                'pred_labels': {method: [] for method in methods},
                'target_correlations': target_correlations,
                }
        for i, result in enumerate(results):
            bounds, largest_bin_results, uniform_prior_empirical_results = result
            attack_results['bounds'].append(bounds)
            # Results for the no ML attacks.
            attack_results['pred_labels']['largest_bin'].append(
                largest_bin_results['pred'])
            attack_results['pred_labels']['uniform_prior_empirical'].\
                append(uniform_prior_empirical_results['pred'])

        attack_results['target_labels'] = target_labels
        #print('target labels', target_labels)

        attack_results['accuracy'] = dict()
        for method in methods:
            pred_labels = attack_results['pred_labels'][method]
            #print(method, 'predicted labels', pred_labels)
            attack_results['accuracy'][method] = \
                accuracy_score(target_labels, pred_labels)
    else:
        # The results have already been aggregated.
        attack_results = results
    print(attack_results['accuracy']['uniform_prior_empirical'])


    print(f'Saving the results to {save_path}...')
    with open(save_path, 'wb') as f:
        attack_results['done'] = True
        pickle.dump(attack_results, f)
