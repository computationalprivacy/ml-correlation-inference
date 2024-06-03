from copulas.multivariate.gaussian import GaussianMultivariate
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


from src.correlations_generator import (compute_accuracy_no_ml,
        compute_empirical_correlations,
        compute_shadow_bounds,
        generate_correlation_matrices)
from src.experiments import init_univariates
from src.helpers.utils import (
        extract_correlation_matrices,
        get_confusion_matrix, 
        get_device,
        get_labels,
        get_other_args,
        predict,
        predict_proba,
        set_torch_seed,
        standardize_dataset,
        train_model,
        train_model_and_extract_features)
from src.meta_model import train_and_evaluate_meta_model
from src.models import MLPTorch
from src.synthetic_data import generate_dataset_from_correlation_matrix


def generate_triplets_aia(n):
    """
    Generates all the unique triplets consisting of a variable X1 followed by a 
    pair of variables X2, X3 (whose order does not matter) that are different 
    from the first.
    """
    assert n >= 3, f'ERROR: Cannot select triplets for {n} columns.'
    triplets = []  
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n, 1):
                if i != j and k != i and k != j:
                    triplets.append( (i, j, k) )
    return triplets


def get_prefix(args):
    prefix = f'ntr-{args.nbr_target_records}' + \
            f'_ns-{args.nbr_shadow_datasets}' +\
            f'_nds-{args.nbr_data_samples}' +\
            f'_ndsbb-{args.nbr_data_samples_bb_aux}'
    if args.shadow_model_type == 'logregdp':
        prefix += f'_eps-{args.epsilon}'
    prefix += f'_nb-{args.nbr_bins}'
    return prefix


def experiment_aia(save_dir, args, seed_repetition):
    save_dir = os.path.join(save_dir, 
            f'smt-{args.shadow_model_type}_mmt-{args.meta_model_type}',
            f'nt-{args.nbr_targets}',
            f'{seed_repetition}')
    print('Save directory', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Path where plots will be saved (if applicable). We create the directory
    # here, rather in the parallelized code, to avoid concurrency issues.
    save_plots_dir = os.path.join(save_dir, 'plots', args.attack_method)
    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)
    # Same here.
    if 'cia' in args.attack_method:
        save_dir_cia = os.path.join(save_dir, 'cia')
        if not os.path.exists(save_dir_cia):
            os.makedirs(save_dir_cia)

    prefix = get_prefix(args)
    save_path = f'{save_dir}/results_{args.attack_method}_{prefix}.pickle'

    # Seeding the sampling of columns.
    np.random.seed(seed_repetition)

    # Load the dataset.
    dataset_path = os.path.join(f'{args.datasets_dir}/{args.dataset_name}/dataset.csv')
    dataset = pd.read_csv(dataset_path)
    assert 'y' in dataset.columns, 'ERROR: The dataset has no `y` variable.'
    total_nbr_columns = len(dataset.columns)
    print(f'Successfully loaded the {args.dataset_name} dataset containing',
            f'{total_nbr_columns} columns and {len(dataset)} samples.\n')
    y_binary = True

    # We run the attack against datasets of 4 columns (including the output
    # variable `y`). We generate all the possible triplets consisting of a
    # sensitive variables (X1) and a pair of two other variables X2 and X3.
    assert args.nbr_columns == 4, 'ERROR: The number of columns used in ' + \
            'the attack against the real datasets should be 4.'
    all_triplets = generate_triplets_aia(total_nbr_columns-1)
    print(f'The dataset yields a total of {len(all_triplets)} unique target',
            'correlations (pairs of variables) coupled with a third variable.')
    # We shuffle the triplets.
    all_triplets = np.random.permutation(all_triplets)

    args_list = []
    attack_results = {'X_cols': [], 'univariates': []}

    # Run the attribute inference attack separately against each target 
    # dataset.
    for ti in range(min(args.nbr_targets, len(all_triplets))):
        seed_target = np.random.randint(10**8)
        triplet = all_triplets[ti]
        X_cols = [dataset.columns[t] for t in triplet]
        target_dataset = dataset[X_cols + ['y']]
        if args.dataset_name in ['fifa19', 'fifa19_v2']:
            # We downsample records from the FIFA19 dataset.
            target_dataset = target_dataset.sample(args.nbr_data_samples,
                    random_state=seed_target)
        target_dataset = standardize_dataset(target_dataset, y_binary)
        X, y = target_dataset[X_cols], target_dataset['y'].astype(int)
        univariates = init_univariates(target_dataset, 
                args.nbr_marginal_bins, y_binary)
        args_list.append( (args, ti, X, y, univariates, seed_target, save_dir) )

        # Saving these to the disk as they might be useful for debugging.
        attack_results['X_cols'].append(X_cols)
        attack_results['univariates'].append(univariates)

    synchronize = (args.shadow_model_type == 'mlptorch' or \
            args.meta_model_type == 'mlptorch') and args.device == 'cuda'
    results = run_attack_parallel(args_list, args.nbr_cores, synchronize)
    mses, all_X1_true, all_X1_pred = [], [], []
    accs, all_binned_X1_true, all_binned_X1_pred = [], [], []
    all_quantiles = []
    target_correlation_matrices = []
    for ti, result in enumerate(results):
        X1_true, X1_pred, quantiles, target_correlation_matrix = result
        #print(f'[{ti+1}], {X1_true[:5]}, {X1_pred[:5]}')
        all_X1_true.append(X1_true)
        all_X1_pred.append(X1_pred)
        # Compute the mean square error.
        mses.append(MSE(X1_true, X1_pred))
        # Compute the accuracy for 3-way classification.
        binned_X1_true = [ map_to_bin(x1, quantiles) for x1 in X1_true ]
        binned_X1_pred = [ map_to_bin(x1, quantiles) for x1 in X1_pred ]
        all_binned_X1_true.append(binned_X1_true)
        all_binned_X1_pred.append(binned_X1_pred)
        accs.append(accuracy_score(binned_X1_true, binned_X1_pred))
        all_quantiles.append(quantiles)
        target_correlation_matrices.append(target_correlation_matrix)
    print(f'Results for attack method={args.attack_method}')
    print(f'MSE: mean={np.mean(mses):.2f}, std={np.std(mses):.2f}')
    print(f'Accuracy (3 classes): mean={np.mean(accs):.1%}, std={np.std(accs):.1%}')

    attack_results.update({'mse': mses,
        'X1_true': all_X1_true,
        'X1_pred': all_X1_pred, 
        'quantiles': quantiles,
        'binned_X1_true': all_binned_X1_true,
        'binned_X1_pred': all_binned_X1_pred,
        'accs': accs,
        'target_correlation_matrices': target_correlation_matrices})

    print(f'Saving the results to {save_path}.')
    with open(save_path, 'wb') as f:
        pickle.dump(attack_results, f)


def map_to_bin(x1, quantiles):
    for i in range(len(quantiles) - 1):
        if quantiles[i] <= x1 < quantiles[i+1]:
            return i
    # Edge case where x1 is equal to the maximum.
    return len(quantiles) - 1


def run_attack_one_target_parallel(args):
    return run_attack_one_target(*args)


def run_attack_one_target(args, ti, X, y, univariates, seed, save_dir):
    """
    args: Experiment parameters (as given via the command line arguments).
    X, y: The target dataset.
    univariates: The univariates of the distribution (fitted on X and y).
    ti: The index of the dataset columns used in the current experiment (for 
        stdout debugging purposes).
    """
    # Data splitting/shuffling.
    np.random.seed(seed+1)
    if args.target_test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                test_size=args.target_test_size)
    else:
        X_train, X_test, y_train, y_test = X, [], y, []
        shuffled_idxs = np.random.permutation(len(X_train))
        X_train, y_train = X_train.iloc[shuffled_idxs], \
                y_train.iloc[shuffled_idxs]
    #print(f'[{ti+1}] Triplet {list(X.columns)}.',
    #        f'Sensitive column {X.columns[0]}.',
    #        'Train and test sizes: ', len(X_train), len(X_test))

    # Store the target correlation matrices.
    train = pd.concat([X_train, y_train], axis=1)
    #print(train.head())
    target_correlation_matrix = train.corr().to_numpy()
    #print(f'[{ti+1}] Target correlation matrix', target_correlation_matrix)
    # These are the correlation constraints which will be used by our attack.
    bounds = target_correlation_matrix[:-1, -1]

    # Train and seed the target model.
    device = get_device(args.device, args.nbr_gpus, ti)
    np.random.seed(seed+2)
    set_torch_seed(args.shadow_model_type, device, seed+2)
    other_args_shadow = get_other_args(args.shadow_model_type,
            args.nbr_columns-1, 2, device, real_dataset=True)
    target_model, acc_train, acc_test = train_model(
            (X_train, y_train, X_test, y_test), args.shadow_model_type, 
            other_args_shadow)
    print(f'[{ti+1}] Target model accuracy: train={acc_train:.1%}',
            f'test={acc_test:.1%}')
    
    # Take the first --nbr_target_records from the training dataset.
    # The choice of records is random because the dataset was shuffled.
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    target_idxs = np.arange(min(args.nbr_target_records, len(X_train)))
    # Moving the dataset to numpy for ease of indexing.
    X_target, y_target = X_train[target_idxs], y_train[target_idxs]

    # The ground-truth sensitive attribute.
    X1_true = X_target[:, 0]
    
    if args.attack_method == 'fredrikson':
        confusion_matrices = {
            'train': get_confusion_matrix(target_model, X_train, y_train),
            'test': get_confusion_matrix(target_model, X_test, y_test)}   
        #print(f'[{ti+1}] Train confusion matrix')
        #print_confusion_matrix(confusion_matrices['train'])
        #print(f'[{ti+1}] Test confusion matrix')
        #print_confusion_matrix(confusion_matrices['test'])
        X1_pred = run_fredrikson_attack(
            target_model,
            univariates,
            X_target[:, 1:],
            y_target,
            confusion_matrices['train'],
            seed+3)
    elif args.attack_method == 'marginal_prior':
        X1_pred = run_marginal_prior_attack(
                univariates[0],
                len(target_idxs),
                seed+3)
    elif args.attack_method == 'csmia':
        X1_pred = run_csmia_attack(
                target_model,
                univariates[0],
                X_target[:, 1:],
                y_target,
                seed+3)
    elif 'cia_aia' in args.attack_method:
        X1_pred = run_correlation_inference_based_attribute_inference_attack(ti,
                target_model,
                univariates,
                X_target[:, 1:],
                y_target,
                bounds,
                len(X_train), # Number of data samples.
                args,
                seed+3,
                device, 
                save_dir,
                X1_true)
    elif args.attack_method == 'copula_base':
        X1_pred = run_copula_base_attack(ti,
            target_correlation_matrix[-1][0], # rho(X1, Y)
            (univariates[0], univariates[-1]), # One-way marginals of X1 and Y.
            y_target,
            seed+3,
            save_dir,
            X1_true
        )
    elif args.attack_method == "yeom":
        X1_pred = run_yeom_attack(
            target_model,
            X_target[:, 1:],
            y_target,
            univariates[0],
            seed+3
        )
    elif args.attack_method == 'cai':
        X1_pred = run_cai_attack(
                target_model,
                X_target[:, 1:],
                y_target,
                univariates[0],
                seed+3)
    elif args.attack_method == 'wcai':
        X1_pred = run_wcai_attack(
                target_model,
                X_target[:, 1:],
                y_target,
                univariates[0],
                seed+3)
    elif 'cia_synth_wcai' in args.attack_method:
        version = args.attack_method[-2:]
        assert version in ['v1', 'v2', 'v3', 'v4'], f"ERROR: Unknown version {version}"
        X1_pred = run_cia_synth_wcai_attack(ti,
                target_model,
                univariates,
                X_target[:, 1:],
                y_target,
                bounds,
                len(X_train), # Number of data samples
                args,
                seed+3,
                device,
                save_dir,
                X1_true,
                version)
    else:
        raise ValueError(f'Invalid --attack_method={args.attack_method}')
    
    # Compute the quantiles of the sensitive attribute X1. By default, we stick
    # to 3 bins.
    X1_name = X.columns[0]
    quantiles = [X[X1_name].min(), 
            X[X1_name].quantile(1/3), 
            X[X1_name].quantile(2/3),
            X[X1_name].max()]
    #print('quantiles', quantiles)
    
    return X1_true, X1_pred, quantiles, target_correlation_matrix


def run_attack_parallel(args_list, nbr_cores, synchronize):
    results = []
    if nbr_cores == 1:
        for args in tqdm(args_list):
            result = run_attack_one_target(*args)
            results.append(result)
    elif synchronize:
        # Running the attack on `nbr_cores` processes at a time, to ensure that
        # the CUDA memory is freed properly. This is slower than launching
        # all processes at the same time because it waits for all `nbr_cores`
        # process to be done.
        for batch_start in range(0, len(args_list), nbr_cores):
            batch_end = min(batch_start + nbr_cores, len(args_list))
            print(f'Grid cells {batch_start}-{batch_end}/{len(args_list)}')
            args_list_batch = args_list[batch_start:batch_end]
            with mp.Pool(nbr_cores) as pool:
                for result in tqdm(pool.imap(run_attack_one_target_parallel, 
                        args_list_batch), total=len(args_list_batch)):
                    results.append(result)
    else:
        with mp.Pool(nbr_cores) as pool:
            for result in tqdm(pool.imap(run_attack_one_target_parallel, 
                    args_list), total=len(args_list)):
                results.append(result)

    return results


def print_confusion_matrix(cm):
    print(f'p(Y=0|f(X)=0)={cm[0][0]:.1%}, p(Y=0|f(X)=1)={cm[0][1]:.1%}',
            f'\np(Y=1|f(X)=0)={cm[1][0]:.1%}, p(Y=1|f(X)=1)={cm[1][1]:.1%}')


def run_marginal_prior_attack(univariate, nbr_samples, seed):
    """
    Sample from the marginal of the sensitive attribute.
    """
    np.random.seed(seed)

    X_preds = univariate.sample(nbr_samples=nbr_samples)
    return X_preds


def run_csmia_attack(target_model, univariate, X_partial_target, y_target,
        seed):
    """
    Runs the Confidence Score-based Model Inversion Attack (CSMIA) by Mehnaz
    et al. (USENIX Security 2022).
    """
    np.random.seed(seed)
    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()
    # The one-way marginal of the sensitive attribute.
    target_bins = univariate.bins

    X1_preds = []
    for i, partial_record in enumerate(X_partial_target):
        # Confidence scores for correctly predicted candidate records.
        cscores_corr_predicted = []
        # Confidence scores for incorrectly predicted candidate records.
        cscores_incorr_predicted = []
        # Iterate over all the possible values of the sensitive attribute. Since
        # the variable is continuous, we use a discretization of the one-way
        # marginal, sampling a value uniformly at random in each bin.
        X1_cands = []
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            #X1_cand = (target_bins[j] + target_bins[j+1] ) / 2 
            candidate_record = np.concatenate([[X1_cand], partial_record]).\
                    reshape(1, -1)
            #print('candidate_record', candidate_record)
            y_pred_proba = predict_proba(target_model, candidate_record)[0]
            y_pred = predict(target_model, candidate_record)[0]
            #print(y_pred_proba, y_pred)
            # The model's confidence for the returned prediction y_pred. Note
            # that y_pred_proba is computed for class #0.
            cscore = max(y_pred_proba, 1-y_pred_proba)
            if y_pred == y_target[i]:
                cscores_corr_predicted.append( (j, cscore) )
            else:
                cscores_incorr_predicted.append( (j, cscore) )
            X1_cands.append(X1_cand)
        if len(cscores_corr_predicted) == 1:
            b_pred = cscores_corr_predicted[0][0]
        elif len(cscores_corr_predicted) > 1:
            M = np.argmax(np.array(cscores_corr_predicted)[:, 1])
            b_pred = cscores_corr_predicted[M][0]
        else:
            assert len(cscores_incorr_predicted) > 0
            m = np.argmin(np.array(cscores_incorr_predicted)[:, 1])
            b_pred = cscores_incorr_predicted[m][0]
        X1_pred = X1_cands[b_pred]
        X1_preds.append(X1_pred)
    return X1_preds


def run_fredrikson_attack(target_model, univariates, X_partial_target, 
        y_target, confusion_matrix, seed):
    np.random.seed(seed)
    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()

    # The one-way marginal of the sensitive attribute.
    target_bins = univariates[0].bins

    X1_preds = []

    for i, partial_record in enumerate(X_partial_target):
        # Iterate over all possible values of the sensitive attribute. Since
        # the one-way marginal is discretized, we sample a value uniformly at
        # random in each bin.
        X1_cands, scores = [], []
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            candidate_record = np.concatenate([[X1_cand], partial_record])
            y_pred = predict(target_model, candidate_record.reshape(1, -1))[0]
            score = confusion_matrix[y_target[i]][y_pred]
            pdf = univariates[0].cdf[j+1] - univariates[0].cdf[j]
            score = score * pdf
            scores.append(score)
            X1_cands.append(X1_cand)
        b_pred = np.argmax(scores)
        X1_pred = X1_cands[b_pred]

        X1_preds.append(X1_pred)
    return X1_preds


def perfect_oracle(X,Y,xi,yi):
    data = pd.concat([X,Y],axis=1)
    x_data = np.concatenate([xi,[yi]])
    #data_set = set(map(tuple,data.to_numpy()))
    #if tuple(x_data) in data_set:
    if x_data in data.to_numpy():
        return 1
    else:
        return 0


def oracle(target_model,candidate_record,y_real):
    y_pred_proba = target_model.predict_proba(candidate_record.reshape(1, -1))
    if y_pred_proba[0, y_real] > 0.5:
        return 1
    else :
        return 0


def run_yeom_attack(target_model, X_partial_targets, y_targets, univariate, seed):
    np.random.seed(seed)
    
    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()
    # The one-way marginal of the sensitive attribute.
    target_bins = univariate.bins
    
    X1_preds = []
    
    for i, partial_record in enumerate(X_partial_targets):
        # Iterate over all possible values of the sensitive attribute.
        X1_cands, scores = [],[]
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            candidate_record = np.concatenate([[X1_cand], partial_record])
            score = univariate.cdf[j+1] - univariate.cdf[j]
            score = score*oracle(target_model,candidate_record, y_targets[i])#perfect_oracle(X_train,y_train,candidate_record,y_pred)#
            scores.append(score)
            X1_cands.append(X1_cand)
        b_pred = np.argmax(scores)
        X1_pred = X1_cands[b_pred]
        X1_preds.append(X1_pred)
    
    return X1_preds


def run_cai_attack(target_model, X_partial_targets, y_targets, univariate, 
        seed):
    np.random.seed(seed)

    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()
    # The one-way marginal of the sensitive attribute.
    target_bins = univariate.bins

    X1_preds = []

    for i, partial_record in enumerate(X_partial_targets):
        # Iterate over all possible values of the sensitive attribute.
        X1_cands, scores = [],[]
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            candidate_record = np.concatenate([[X1_cand], partial_record])
            score = target_model.predict_proba(
                    candidate_record.reshape(1, -1))[0, y_targets[i]]
            #score = univariate.cdf[j+1] - univariate.cdf[j]
            scores.append(score)
            X1_cands.append(X1_cand)
        b_pred = np.argmax(scores)
        X1_pred = X1_cands[b_pred]
        X1_preds.append(X1_pred)

    return X1_preds


def run_wcai_attack(target_model, X_partial_targets, y_targets, univariate, 
        seed):
    np.random.seed(seed)

    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()
    # The one-way marginal of the sensitive attribute.
    target_bins = univariate.bins

    X1_preds = []

    for i, partial_record in enumerate(X_partial_targets):
        # Iterate over all possible values of the sensitive attribute.
        X1_cands, scores = [],[]
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            candidate_record = np.concatenate([[X1_cand], partial_record])
            score = univariate.cdf[j+1] - univariate.cdf[j]
            score = score * target_model.predict_proba(
                    candidate_record.reshape(1, -1))[0, y_targets[i]]
            scores.append(score)
            X1_cands.append(X1_cand)
        b_pred = np.argmax(scores)
        X1_pred = X1_cands[b_pred]
        X1_preds.append(X1_pred)

    return X1_preds


def run_copula_base_attack(ti, corr_x1y, univariates, y_targets, seed, 
        save_dir, X1_true):
    #This function will run a simple copula_based attack. 
    #Under the knowledge of the correlation between X1 and Y, and the marginal 
    # distributions of X1 and Y, we compute directly P(X1 |Y)
    np.random.seed(seed)
    
    shifted_corr_x1y = compute_shadow_bounds(np.array([corr_x1y]),
            univariates,
            False,
            nbr_columns=2)
    shifted_matrix =  np.array([
        [1, shifted_corr_x1y],
        [shifted_corr_x1y, 1]])
    X, y, _, _ = generate_dataset_from_correlation_matrix(
                    shifted_matrix, univariates, 100000)
    dataset = pd.concat([X, y], axis=1) 
    X1_preds = []
    for i, y in enumerate(y_targets):
        matches = dataset.loc[dataset['y'] == y]['x1']
        X1_pred = matches.mean()
        X1_preds.append(X1_pred)
    return X1_preds


# AIA using CIA + synthetic data + 
# V1: argmax_g p(x1^g|x2,...,xn-1)
# V2: argmax_g p(x1^g|x2,...,xn-1,y)
# V3: argmax_g p(x1^g|x2,...,xn-1) * Vy^g, where Vy^g denotes the model's confidence
# for the partial record's label.
# V4: argmax_g p(x1^g|x2,...,xn-1,y) * Vy^g 
def run_cia_synth_wcai_attack(ti,
        target_model,
        univariates,
        X_partial_target,
        y_target,
        bounds,
        nbr_data_samples,
        args,
        seed,
        device,
        save_dir,
        X1_true,
        version):
    #first, take the return of V1
    bounds, uniform_prior_empirical_results, access_to_model_results = \
            run_correlation_inference_attack(ti,
                    target_model,
                    univariates,
                    bounds,
                    nbr_data_samples,
                    args,
                    seed,
                    device,
                    save_dir+'/cia')
    
    np.random.seed(seed)
    
    # Step 1. Same as us
    shadow_bounds = np.mean(list(bounds[1].values()), axis=0)
    correlation_matrices = generate_correlation_matrices(
            nbr_columns=4,
            nbr_trials=1000,
            bounds=shadow_bounds,
            lengths=None,
            constraints_scenario='column',
            balanced=False)
    datasets = [generate_dataset_from_correlation_matrix(
            correlation_matrix, univariates, nbr_data_samples)
            for correlation_matrix in correlation_matrices]
    datasets = [pd.concat((X, y), axis=1) for X, y, _, _ in datasets]
    empirical_correlations = [dataset.corr().to_numpy() for dataset in datasets]
   
    # Step 2. Same as us
    attack_results = access_to_model_results
    attack_labels = {pair: result[0] for pair, result in attack_results.items()}
    #print(attack_labels)
    datasets = [d for d, corr in zip(datasets, empirical_correlations)
            if is_match(corr, args.nbr_bins, attack_labels)]
    #print('Number of matching datasets', len(datasets))

    if len(datasets) == 0:
        print(f'[{ti+1}] Did not find any matching matrix.')
        return univariates[0].sample(nbr_samples=len(y_target))
    # Putting together all the datasets to form a single dataset.
    qd = pd.concat(datasets, axis=0)

    #Step 3 : training of a model to predict x1 based on x2 .. xn (and 
    # optionally y).
    if version == 'v1' or version == 'v3':
        X_train = qd.drop(columns=['x1', 'y']).to_numpy()
    elif version == 'v2' or version == 'v4':
        X_train = qd.drop(columns=['x1']).to_numpy()
    else:
        raise ValueError(f'ERROR: Unknown version {version}')
    y_train = qd[['x1']].to_numpy()

    model = LinearRegression().fit(X_train, y_train)

    if version == 'v1' or version == 'v3':
        X1_imputation = model.predict(X_partial_target)
    elif version == 'v2' or version == 'v4':
        X1_imputation = model.predict(np.concatenate([X_partial_target, y_target.reshape(-1, 1)], axis=1))
    else:
        raise ValueError(f'ERROR: Unknown version {version}')

    if version == 'v1' or version == 'v2':
        return X1_imputation
    
    # It is faster to query the model on the CPU.
    if isinstance(target_model, MLPTorch):
        target_model.to_cpu()
    # The one-way marginal of the sensitive attribute.
    target_bins = univariates[0].bins 

    X1_preds = []
    for i, partial_record in enumerate(X_partial_target):
        # Iterate over all the possible values of the sensitive attribute. Since
        # the variable is continuous, we use a discretization of the one-way
        # marginal, sampling a value uniformly at random in each bin.
        scores = []
        X1_cands = []
        for j in range(len(target_bins)-1):
            bin_start, bin_end = target_bins[j], target_bins[j+1]
            u = np.random.random()
            X1_cand = u * (bin_end - bin_start) + bin_start
            candidate_record = np.concatenate([[X1_cand], partial_record]).\
                    reshape(1, -1)
            y_pred_proba = predict_proba(target_model, candidate_record)[0]
            if y_target[i] == 0:
                cscore = y_pred_proba
            else:
                cscore = 1 - y_pred_proba
            d = np.abs(X1_imputation[i]-X1_cand)
            scores.append(cscore*math.e**(-d*d/2))
            X1_cands.append(X1_cand)
        b_pred = np.argmax(scores)
        X1_preds.append(X1_cands[b_pred])
    return X1_preds


def run_correlation_inference_based_attribute_inference_attack(ti,
        target_model,
        univariates,
        X_partial_target,
        y_target,
        bounds,
        nbr_data_samples,
        args,
        seed,
        device,
        save_dir,
        X1_true):
    bounds, uniform_prior_empirical_results, access_to_model_results = \
            run_correlation_inference_attack(ti,
                    target_model,
                    univariates,
                    bounds,
                    nbr_data_samples,
                    args,
                    seed,
                    device,
                    save_dir+'/cia')

    # Step 1. We combine the shadow bounds and use them to generate candidate 
    # correlation matrices. For each of these matrices, we generate synthetic 
    # data and compute the empirical correlations.
    # Step 2. We select only the datasets having empirical correlation matrices
    # with rho(Xi, Xj) inside the bin inferred by our attack, for all i and j.
    # Step 3. We run the attribute inference attack.
    np.random.seed(seed) 
    # Step 1.
    shadow_bounds = np.mean(list(bounds[1].values()), axis=0)
    correlation_matrices = generate_correlation_matrices(
            nbr_columns=4,
            nbr_trials=1000,
            bounds=shadow_bounds,
            lengths=None,
            constraints_scenario='column',
            balanced=False)
    datasets = [generate_dataset_from_correlation_matrix(
            correlation_matrix, univariates, nbr_data_samples)
            for correlation_matrix in correlation_matrices]
    datasets = [pd.concat((X, y), axis=1) for X, y, _, _ in datasets]
    empirical_correlations = [dataset.corr().to_numpy() for dataset in datasets]

    # Step 2.
    if args.attack_method == 'cia_aia_model_less':
        attack_results = uniform_prior_empirical_results
    else:
        attack_results = access_to_model_results
    attack_labels = {pair: result[0] for pair, result in attack_results.items()}
    #print(attack_labels)
    datasets = [d for d, corr in zip(datasets, empirical_correlations)
            if is_match(corr, args.nbr_bins, attack_labels)]
    #print('Number of matching datasets', len(datasets))

    if len(datasets) == 0:
        print(f'[{ti+1}] Did not find any matching matrix.')
        return univariates[0].sample(nbr_samples=len(y_target))
    # Putting together all the datasets to form a single dataset.
    qd = pd.concat(datasets, axis=0)

    # Step 3. Determine the value of x1 that maximizes p(x1 | x2, x3, y). 
    
    # Length of the bins for each column.
    b2, b3 = get_bin_length(univariates[1]), get_bin_length(univariates[2])
    X1_preds = []
    for i, partial_record in enumerate(X_partial_target):
        (x2, x3), y = partial_record, y_target[i]
        # By how much to multiply the bin size; by default we search around
        # the record and then we progressively increase the resolution until
        # we find at least one matching record.
        m2, m3 = 2, 2
        found = False
        while not found:
            # How many records match the partial record?
            matches = qd.loc[
                (x2 - m2 * b2 <= qd['x2']) & (qd['x2'] < x2 + m2 * b2) &
                (x3 - m3 * b3 <= qd['x3']) & (qd['x3'] < x3 + m3 * b3) &
                (qd['y'] == y)
                ]['x1']
            if len(matches) > 0:
                found = True
            else: # If no match is found, increase the resolution.
                m2 += 0.5
                m3 += 0.5
        X1_pred = matches.mean()
        X1_preds.append(X1_pred) 
    return X1_preds


def get_bin_length(univariate):
    return univariate.bins[1] - univariate.bins[0]


def is_match(correlation_matrix, nbr_bins, target_labels):
    """
    Checks that the correlation matrix matches the target labels at the 
    positions given the by the keys of `target_labels` (a dictionary of 
    pair: label).
    """
    is_match = True
    for i, j in target_labels:
       label = get_correlation_bin(correlation_matrix[i][j], nbr_bins) 
       if label != target_labels[(i, j)]:
           is_match = False
    return is_match


def get_correlation_bin(corr, nbr_bins):
    return int( (corr + 1) * nbr_bins / 2 )


def run_correlation_inference_attack(ti, target_model, univariates, 
        bounds, nbr_data_samples, args, seed, device, save_dir):
    """
    Our attack first extracts the correlations of (X1, X2), (X2, X3) and 
    (X1, X3) from the model. It then predicts the most likely bin according to 
    the joint probability given by the copulas parametrized by the inferred 
    correlation matrix.

    We run this attack in the column scenario. In this scenario, the attacker 
    knows rho(X1, Y), rho(X2, Y) and rho(X3, Y).

    A subtletly is that by default our attack infers the correlation between 
    the first two columns. 
    The shadow correlation matrix generation procedure takes this into account, 
    balancing the values generated for the target correlation rho(X1, X2) when 
    --balanced_train=True.
    To maintain this behavior, we need to bring the target correlations into 
    the first two positions before generating the shadow correlation matrices.
    We then permute them back to the original order before resuming the attack.
    """
    prefix = get_prefix(args)
    save_path = os.path.join(save_dir, f'{ti}_{prefix}.pickle')
    if os.path.exists(save_path):
        print(f'[{ti+1}] Found existing results for the correlation inference attack in {save_path}.')
        with open(save_path, 'rb') as f:
            attack_results = pickle.load(f)
        return attack_results['results']
    else:
        print(f'[{ti+1}] Did not find existing results for the correlation inference attack in {save_path}.')

    pairs = [(0, 1), (0, 2), (1, 2)]
    permutations = [ [0, 1, 2, 3], [0, 2, 1, 3], [2, 0, 1, 3] ]
    inverse_permutations = [ [0, 1, 2, 3], [0, 2, 1, 3], [1, 2, 0, 3] ]
    seeds = [seed + 10, seed + 20, seed + 30]
    
    # We will re-use these for the attribute inference attack.
    all_shadow_bounds = dict()
    uniform_empirical_prior_results = dict()
    access_to_model_results = dict()
    for pi, (i, j) in enumerate(pairs):
        np.random.seed(seeds[pi])
        # Permute the bounds so as to bring the target pair into the first two
        # positions.
        permuted_bounds = bounds[permutations[pi][:-1]]
        # Permute the univariates.
        permuted_univariates = [univariates[k] for k in permutations[pi]]
        shadow_permuted_bounds = compute_shadow_bounds(
                permuted_bounds,
                permuted_univariates,
                args.balanced_train,
                nbr_data_samples=nbr_data_samples)
        #print('Shadow permuted bounds', shadow_permuted_bounds)
        shadow_correlation_matrices = generate_correlation_matrices(
                4, # Number of columns.
                args.nbr_shadow_datasets+1,
                shadow_permuted_bounds,
                None, # No lenghts argument.
                'column',
                args.balanced_train)

        # Invert the permutation before generating the datasets.
        sigma = inverse_permutations[pi]
        shadow_correlation_matrices = [c[sigma][:, sigma]
                for c in shadow_correlation_matrices]
        shadow_bounds = shadow_permuted_bounds[inverse_permutations[pi][:-1]]

        attack_correlation_matrix = shadow_correlation_matrices[-1]
        shadow_correlation_matrices = shadow_correlation_matrices[:-1]

        shadow_datasets = [generate_dataset_from_correlation_matrix(
            shadow_correlation_matrix, univariates, nbr_data_samples)
            for shadow_correlation_matrix in shadow_correlation_matrices]
        # The shadow labels are derived from correlations measured empirically 
        # on the shadow datasets.
        empirical_correlation_matrices = extract_correlation_matrices(
            [X_train for X_train, _, _, _ in shadow_datasets])
        shadow_labels = get_labels(empirical_correlation_matrices, i, j, 
                args.nbr_bins, args.meta_model_type)

        # Generate the attack dataset.
        X_attack, _, _, _ = generate_dataset_from_correlation_matrix(
                attack_correlation_matrix, 
                univariates, 
                args.nbr_data_samples_bb_aux)

        # Run the attack which is not using access to the model.
        np.random.seed(seeds[pi]+1)
        # Use the permuted constraints.
        correlation_matrices = generate_correlation_matrices(
                4, # Number of columns
                args.nbr_shadow_datasets,
                permuted_bounds,
                None,
                'column',
                args.balanced_train)
        # The predictions relate to the first pair, after the permutation.
        _, upe_results = compute_accuracy_no_ml(correlation_matrices, 
                args.nbr_bins)

        # Run the model-based attack (black-box only).
        np.random.seed(seeds[pi]+2)
        set_torch_seed(args.shadow_model_type, device, seeds[pi]+2)
        other_args_shadow = get_other_args(args.shadow_model_type, 3, 2,
                device, real_dataset=True)

        model_predictions = []
        for dataset in shadow_datasets:
            features, _ = train_model_and_extract_features(
                    dataset,
                    args.shadow_model_type,
                    other_args_shadow,
                    X_attack,
                    args.nbr_significant_figures)
            model_predictions.append(features['model_predictions'])

        other_args_meta = get_other_args(args.meta_model_type,
            len(model_predictions[0][0]), args.nbr_bins, device, meta=True)
        
        # Extract features from the target model.
        target_predictions = predict_proba(target_model, X_attack,
            args.nbr_significant_figures).reshape(1, -1)

        # Train the meta-model.
        accs, meta_model, scaler = train_and_evaluate_meta_model(
            model_predictions,
            shadow_labels,
            args.meta_model_type,
            False, # Do not use k-fold cross validation.
            other_args_meta,
            verbose=args.verbose)

        if scaler is not None:
            tpredictions = scaler.transform(target_predictions)
        else:
            tpredictions = target_predictions
        #print(tpredictions)
        pred_label = predict(meta_model, tpredictions)[0]

        uniform_empirical_prior_results[(i, j)] = upe_results['pred'], \
                upe_results['acc']
        access_to_model_results[(i, j)] = pred_label, accs
        all_shadow_bounds[(i, j)] = shadow_bounds

    print(f'[{ti+1}] Done executing the correlation inference attack. Saving the results to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump({'results': ( (bounds, all_shadow_bounds), 
            uniform_empirical_prior_results, 
            access_to_model_results )}, f)

    #print(all_shadow_bounds)
    print(f'[{ti+1}] upe results', uniform_empirical_prior_results)
    print(f'[{ti+1}] cia results', access_to_model_results)
    print(f'[{ti+1}] bounds', bounds)
    print(f'[{ti+1}] shadow bounds', shadow_bounds)
    return (bounds, all_shadow_bounds), uniform_empirical_prior_results, \
            access_to_model_results

