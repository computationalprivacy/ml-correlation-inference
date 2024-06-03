from collections import defaultdict
import pandas as pd
import pickle
import numpy as np
import time

from src.correlations_generator import (generate_correlation_matrices,
        compute_accuracy_no_ml, 
        compute_empirical_correlations, 
        compute_shadow_bounds)
from src.helpers.utils import (compute_bounds_from_constraints,
        extract_correlation_matrices,
        get_labels, 
        get_model_weights, 
        get_other_args,
        predict, 
        set_torch_seed,
        train_model,
        train_model_and_extract_features)
from src.meta_model import train_and_evaluate_meta_model
from src.synthetic_data import generate_dataset_from_correlation_matrix


def correlation_inference_attack_synthetic(univariates,
        nbr_columns,
        nbr_shadow_datasets,
        nbr_data_samples,
        test_size,
        nbr_data_samples_bb_aux,
        shadow_model_type,
        device,
        meta_model_type,
        nbr_bins,
        bounds,
        lengths,
        constraints_scenario,
        balanced,
        seed,
        use_kfold,
        target_dataset=None,
        nbr_significant_figures=-1,
        verbose=False,
        epsilon=1.0,
        same_seed=False):
    # Initialize the random number generator at the beginning.
    np.random.seed(seed)

    # Generate the correlation matrices once for all the attack models.
    correlation_matrices = generate_correlation_matrices(nbr_columns,
            nbr_shadow_datasets, bounds, lengths, constraints_scenario, 
            balanced)
    #print(correlation_matrices[:2])

    # We only target the correlation between the first two variables.
    labels = get_labels(correlation_matrices, 0, 1, nbr_bins, meta_model_type)
    if verbose:
        count_per_class = [np.sum(labels==b) for b in range(nbr_bins)]
        print('Seed', seed, 'Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(nbr_bins)]))


    # Generate the datasets once for all the attack models relying on machine
    # learning, regardless of the models used.
    datasets = [generate_dataset_from_correlation_matrix(
                correlation_matrix, univariates, nbr_data_samples, test_size)
                for correlation_matrix in correlation_matrices]
    if verbose:
        print('Size of shadow datasets train and test:', 
                len(datasets[0][0]), len(datasets[0][2]))

    # Generate the synthetic datasets (same randomness regardless of the
    # target model used).
    np.random.seed(seed+1)
    X_attack, _, _, _ = generate_dataset_from_correlation_matrix(
            generate_correlation_matrices(nbr_columns, 1, bounds, lengths, 
                constraints_scenario, balanced)[0],
            univariates, nbr_data_samples_bb_aux)

    # Run the attack which is not using ML.
    np.random.seed(seed)
    largest_bin_results, uniform_empirical_prior_results = \
            compute_accuracy_no_ml(correlation_matrices, nbr_bins)
    if verbose:
        print('seed', seed, 'largest_bin', largest_bin_results['pred'], 
                largest_bin_results['acc'])
        print('seed', seed, 'unif emp prior', uniform_empirical_prior_results)
    # Run the attack against ML models. 
    # Seed again so that the results are the same even if the attacks are run 
    # separately. 
    np.random.seed(seed)
    set_torch_seed(shadow_model_type, device, seed)
    other_args_shadow = get_other_args(shadow_model_type, nbr_columns-1, 2,
            device)

    model_features = defaultdict(list)
    #print('Training the shadow models and extracting their features')
    times = []
    accs_train, accs_test = [], []
    acc_target_train, acc_target_test = None, None
    #print('Same seed inside', same_seed)

    for i, dataset in enumerate(datasets):
        #if i % 500 == 0:
        #    print(i)
        #print(f'Training model and extracting features for the {i+1}-th dataset') 
        #start_time = time.time()
        if same_seed:
            #print('I am here 1')
            np.random.seed(seed)
            set_torch_seed(shadow_model_type, device, seed)
        features, (acc_train, acc_test) = train_model_and_extract_features(
                dataset,
                shadow_model_type,
                other_args_shadow,
                X_attack,
                nbr_significant_figures,
                epsilon=epsilon,
                same_seed=same_seed)
        accs_train.append(acc_train)
        accs_test.append(acc_test)
        #times.append(time.time() - start_time)
        #print(f'Average training time: {np.mean(times):.1}secs')
        for fname, fs in features.items():
            if fs is None:
                continue
            model_features[fname].append(features[fname])

    access_to_model_results = dict()
    #print(model_features)
    for fname in model_features:
        #print(fname, model_features[fname][0])
        other_args_meta = get_other_args(meta_model_type,
                len(model_features[fname][0][0]), nbr_bins, device, meta=True)
        if target_dataset is None:
            access_to_model_results[fname] = train_and_evaluate_meta_model(
                    model_features[fname],
                    labels,
                    meta_model_type,
                    use_kfold,
                    other_args_meta,
                    verbose)[0]
        else:
            #print(f'Training model and extracting features for the target dataset')
            if same_seed:
                #print('I am here')
                np.random.seed(seed)
                set_torch_seed(shadow_model_type, device, seed)
            target_features, (acc_train, acc_test) = \
                    train_model_and_extract_features(
                        target_dataset,
                        shadow_model_type,
                        other_args_shadow,
                        X_attack,
                        nbr_significant_figures, 
                        epsilon=epsilon,
                        same_seed=same_seed) #adding dp
            acc_target_train = acc_train
            acc_target_test = acc_test
            accs, meta_model, scaler = train_and_evaluate_meta_model(
                    model_features[fname],
                    labels,
                    meta_model_type,
                    use_kfold,
                    other_args_meta,
                    verbose=verbose)
            if scaler is not None:
                tfeatures = scaler.transform(target_features[fname])
            else:
                tfeatures = target_features[fname]
            pred_label = predict(meta_model, tfeatures)[0]
            if verbose:
                print('seed', seed, 'fname', fname, 'pred_label', pred_label, 
                        'accs', accs)
            access_to_model_results[fname] = pred_label, accs
    return bounds, largest_bin_results, uniform_empirical_prior_results, \
            access_to_model_results, \
            (accs_train, accs_test, acc_target_train, acc_target_test)


def correlation_inference_attack_synthetic_mitigations(univariates,
        nbr_columns,
        nbr_shadow_datasets,
        nbr_data_samples,
        nbrs_data_samples_bb_aux,
        shadow_model_type,
        device,
        meta_model_type,
        nbr_bins,
        bounds,
        constraints_scenario,
        balanced,
        seed,
        target_dataset,
        nbrs_significant_figures,
        verbose=False):
    """
    A pipeline to efficiently evaluate the robustness of the correlation
    inference attack to mitigations. The idea is to train the shadow models
    only once. For each number of significant figures and/or number of 
    queries, we then feed the corresponding features to the meta-classifier.

    We simplify the code of `correlation_inference_attack_synthetic` by 
    removing some of the arguments.
    """
    # Initialize the random number generator at the beginning.
    np.random.seed(seed)

    # Generate the correlation matrices once for all the attack models.
    correlation_matrices = generate_correlation_matrices(nbr_columns,
            nbr_shadow_datasets, bounds, None, constraints_scenario, 
            balanced)

    # We only target the correlation between the first two variables.
    labels = get_labels(correlation_matrices, 0, 1, nbr_bins, meta_model_type)
    if verbose:
        count_per_class = [np.sum(labels==b) for b in range(nbr_bins)]
        print('Seed', seed, 'Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(nbr_bins)]))


    # Generate the datasets once for all the attack models relying on machine
    # learning, regardless of the models used.
    datasets = [generate_dataset_from_correlation_matrix(
                correlation_matrix, univariates, nbr_data_samples)
                for correlation_matrix in correlation_matrices]

    # Generate the synthetic datasets (same randomness regardless of the
    # target model used).
    np.random.seed(seed+1)
    X_attack, _,  _, _ = generate_dataset_from_correlation_matrix(
            generate_correlation_matrices(nbr_columns, 1, bounds, None, 
                constraints_scenario, balanced)[0],
            univariates, 
            nbrs_data_samples_bb_aux[-1] # Use the largest number.
            )

    # Run the attack against ML models. 
    # Seed again so that the results are the same even if the attacks are run 
    # separately. 
    np.random.seed(seed)
    set_torch_seed(shadow_model_type, device, seed)
    other_args_shadow = get_other_args(shadow_model_type, nbr_columns-1, 2,
            device)

    model_predictions = np.zeros((len(datasets), len(X_attack)))
    #print('Training the shadow models and extracting their features')
    for i, dataset in enumerate(datasets):
        features, _ = train_model_and_extract_features(dataset,
                    shadow_model_type,
                    other_args_shadow,
                    X_attack,
                    nbrs_significant_figures[-1] # Use the largest number.
                    )
        assert 'model_predictions' in features, f'ERROR: No model predictions.'
        model_predictions[i] = features['model_predictions']
    #print('Model predictions shape', model_predictions.shape)

    # Results for varying number of significant digits and for varying
    # number of queries, respectively.
    access_to_model_results = [ [], [] ]

    target_features, _ = train_model_and_extract_features(
            target_dataset,
            shadow_model_type,
            other_args_shadow,
            X_attack,
            nbrs_significant_figures[-1])
    target_features = target_features['model_predictions']
    #print('Target features shape', target_features.shape)

    # Results for varying number of significant digits and for varying
    # number of queries, respectively.
    access_to_model_results = [[], []]

    # When varying the number of significant digits, we use 100 queries just 
    # like in the base experiment (where we set --nbr_data_samples_bb_aux=100).
    assert len(X_attack) >= 100

    #print('Running the attack for different number of significant figures:',
    #        nbrs_significant_figures)
    for nbr_significant_figures in nbrs_significant_figures:
        shadow_features = model_predictions[:, :100]
        tfeatures = target_features[:, :100]
        #print('Shapes', shadow_features.shape, tfeatures.shape)
        if nbr_significant_figures >= 0:
            shadow_features = np.round(shadow_features, nbr_significant_figures)
            tfeatures = np.round(tfeatures, nbr_significant_figures)
        other_args_meta = get_other_args(meta_model_type, 100, nbr_bins, 
                device, meta=True)
        #print(nbr_significant_figures, shadow_features[:5])
        np.random.seed(seed)
        set_torch_seed(meta_model_type, device, seed)
        pred_label = train_meta_model_and_predict(shadow_features, tfeatures, 
                labels, meta_model_type, other_args_meta, verbose) 
        access_to_model_results[0].append(pred_label)

    #print('Running the attack for different number of queries:',
    #        nbrs_data_samples_bb_aux)
    for nbr_data_samples_bb_aux in nbrs_data_samples_bb_aux:
        assert nbr_data_samples_bb_aux <= len(model_predictions[0])
        # No rounding, but using different number of features.
        other_args_meta = get_other_args(meta_model_type, 
                nbr_data_samples_bb_aux, nbr_bins, device, meta=True)
        shadow_features = model_predictions[:, :nbr_data_samples_bb_aux]
        tfeatures = target_features[:, :nbr_data_samples_bb_aux]
        #print(nbr_data_samples_bb_aux, len(shadow_features[0]))
        np.random.seed(seed)
        set_torch_seed(meta_model_type, device, seed)
        pred_label = train_meta_model_and_predict(shadow_features, tfeatures, 
                labels, meta_model_type, other_args_meta, verbose)
        access_to_model_results[1].append(pred_label) 

    return bounds, access_to_model_results


def train_meta_model_and_predict(shadow_features, target_features, labels, 
        meta_model_type, other_args_meta, verbose):
    #print('Shadow features size', shadow_features.shape, 
    #        'Target features size', target_features.shape)
    _, meta_model, scaler = train_and_evaluate_meta_model(
            shadow_features,
            labels,
            meta_model_type,
            False, # use_kfold
            other_args_meta,
            verbose)
    if scaler is not None:
        target_features = scaler.transform(target_features)
    pred_label = predict(meta_model, target_features)[0]
    return pred_label


def correlation_inference_attack(
        target_correlation,
        univariates, 
        nbr_columns, 
        nbr_shadow_datasets,
        nbr_data_samples, 
        nbr_data_samples_bb_aux, 
        shadow_model_type,
        device,
        meta_model_type,
        nbr_bins,
        bounds, 
        constraints_scenario,
        balanced,
        seed, 
        use_kfold,
        target_dataset=None,
        nbr_significant_figures=-1,
        verbose=False):
    #start_time = time.time()
    # Initialize the random number generator at the beginning.
    np.random.seed(seed)
    # The datasets are generated from shadow constraints that can be slightly
    # different from the original constraints, because the one-way marginals
    # are not necessarily distributed similarly to N(0,1). 
    shadow_bounds = compute_shadow_bounds(bounds, univariates, balanced,
            nbr_data_samples=nbr_data_samples)
    #shadow_bounds = bounds
    #print(bounds, shadow_bounds)
    
    shadow_correlation_matrices = generate_correlation_matrices(
            nbr_columns, nbr_shadow_datasets, shadow_bounds, None,
            constraints_scenario, balanced)
    #print('Time to compute the shadow bounds: ', time.time()-start_time)

    #start_time = time.time()
    # Generate the datasets once for all the attack models relying on machine
    # learning, regardless of the models used.
    shadow_datasets = [generate_dataset_from_correlation_matrix(
        shadow_correlation_matrix, univariates, nbr_data_samples) 
        for shadow_correlation_matrix in shadow_correlation_matrices]
    #print('Time to sample the shadow datasets', time.time()-start_time)
    
    #datasets_for_shifted = [X_train.merge(y_train,how='inner', left_index=True, right_index=True) for (X_train,y_train,_,_) in datasets]
    
    # The shadow labels are derived from correlations measured empirically on
    # the shadow datasets.
    empirical_correlation_matrices = extract_correlation_matrices(
            [X_train for X_train, _, _, _ in shadow_datasets])
    shadow_labels = get_labels(empirical_correlation_matrices, 0, 1, nbr_bins, 
            meta_model_type)
    
    np.random.seed(seed)
    X_attack, _, _, _ = generate_dataset_from_correlation_matrix(
            generate_correlation_matrices(nbr_columns, 1, shadow_bounds, 
                None, constraints_scenario, balanced)[0],
            univariates, nbr_data_samples_bb_aux)
    
    #print('Running the model-less attack.')
    # Run the attack which is not using ML.
    np.random.seed(seed)
    # This attack is agnostic to the one-way marginals of the target dataset.
    # We thus use the constraints derived from the original (target) dataset.
    correlation_matrices = generate_correlation_matrices(nbr_columns,
            nbr_shadow_datasets, bounds, None, constraints_scenario,
            balanced)
    largest_bin_results, uniform_empirical_prior_results = \
            compute_accuracy_no_ml(correlation_matrices, nbr_bins)
    print('Target correlation', target_correlation, 
            'Original bounds', bounds, 
            compute_bounds_from_constraints(bounds[0], bounds[1]), 
            'Largest bin results', largest_bin_results['acc'], 
            largest_bin_results['pred'], 
            [d[0,1] for d in largest_bin_results['distribution']],
            'Uniform prior empirical results', uniform_empirical_prior_results)
    # Run the attack against ML models. 
    # Seed again so that the results are the same even if the attacks are run 
    # separately. 
    np.random.seed(seed)
    set_torch_seed(shadow_model_type, device, seed)
    other_args_shadow = get_other_args(shadow_model_type, nbr_columns-1, 2, 
            device, real_dataset=True)

    model_features = defaultdict(list)
    print('Training the shadow models and extracting their features')
    for i, dataset in enumerate(shadow_datasets):
        #if i % 500 == 0:
        #    print(i)
        #print(f'Training model and extracting features for the {i+1}-th dataset')
        #start_time = time.time()
        features, _ = train_model_and_extract_features(dataset, 
                    shadow_model_type,
                    other_args_shadow, 
                    X_attack, 
                    nbr_significant_figures)
        #times.append(time.time() - start_time)
        #print(f'Average training time: {np.mean(times):.1}secs')
        for fname, fs in features.items():
            if fs is None:
                continue
            model_features[fname].append(features[fname])
    
   
    access_to_model_results = dict()
    #print(model_features)
    for fname in model_features:
        #print(fname, model_features[fname][0])
        other_args_meta = get_other_args(meta_model_type, 
                len(model_features[fname][0][0]), nbr_bins, device, meta=True)  
        #print(f'Training model and extracting features for the target dataset')
        target_features, _ = train_model_and_extract_features(
                        target_dataset,
                        shadow_model_type, 
                        other_args_shadow, 
                        X_attack, 
                        nbr_significant_figures)
        accs, meta_model, scaler = train_and_evaluate_meta_model(
                model_features[fname], 
                shadow_labels, 
                meta_model_type, 
                use_kfold, 
                other_args_meta, 
                verbose=verbose)
        if scaler is not None:
            tfeatures = scaler.transform(target_features[fname])
        else:
            tfeatures = target_features[fname]
        pred_label = predict(meta_model, tfeatures)[0]
        #print(pred_label)
        access_to_model_results[fname] = pred_label, accs
    return (bounds, shadow_bounds), largest_bin_results, \
            uniform_empirical_prior_results, \
            access_to_model_results


def correlation_extraction(univariates, target_dataset,
        shadow_model_type, nbrs_queries, seed, device):
    # Train the target model.
    #print(seed)
    np.random.seed(seed)
    set_torch_seed(shadow_model_type, device, seed)
    nbr_columns = len(univariates)
    other_args = get_other_args(shadow_model_type, nbr_columns-1, 2,
            device, real_dataset=True)
    target_model, acc_train, acc_test = train_model(target_dataset, 
            shadow_model_type, other_args, verbose=False)
    #print(type(target_model))

    # Sample the query dataset.
    dataset = dict()
    for i in range(len(univariates)-1):
        # We query the model.
        dataset[f'x{i+1}'] = univariates[i].sample(nbr_samples=nbrs_queries[-1])
    dataset = pd.DataFrame(dataset)
    #print(dataset)

    # Pass the query dataset through the model.
    y = predict(target_model, dataset)
    dataset['y'] = y

    # For each number of queries, compute the correlations based on that many
    # records.
    results = {'ours': []}
    for nbr_queries in nbrs_queries:
        y_sum = dataset[:nbr_queries]['y'].sum()
        if y_sum == 0 or y_sum == nbr_queries:
            print(f'Found an edge case where all ys are equal to {int(y_sum>0)} ({nbr_queries} queries)')
            rho_pred = np.random.uniform(size=nbr_columns-1)*2-1
        else:
            correlation_matrix = dataset[:nbr_queries].corr().to_numpy()
            rho_pred = correlation_matrix[-1, :-1]
            assert len(rho_pred) == 3
        results['ours'].append(rho_pred)

    # The random guess is independent of the number of queries.
    random_guess = np.random.uniform(size=nbr_columns-1)*2-1
    results['random_guess'] = [random_guess] * len(nbrs_queries)
    return results


def correlation_inference_attack_synthetic_model_less_only(
        nbr_columns,
        nbr_shadow_datasets,
        meta_model_type,
        nbr_bins,
        bounds,
        constraints_scenario,
        balanced,
        seed,
        verbose=False):
    # Initialize the random number generator at the beginning.
    np.random.seed(seed)

    # Generate the correlation matrices once for all the attack models.
    correlation_matrices = generate_correlation_matrices(nbr_columns,
            nbr_shadow_datasets, bounds, None, constraints_scenario, 
            balanced)
    #print(correlation_matrices[:2])
    #print(len(correlation_matrices))

    # We only target the correlation between the first two variables.
    labels = get_labels(correlation_matrices, 0, 1, nbr_bins, meta_model_type)
    if verbose:
        count_per_class = [np.sum(labels==b) for b in range(nbr_bins)]
        print('Seed', seed, 'Distribution over the target labels. ', ', '.\
            join([f'{b}: {count_per_class[b]}' for b in range(nbr_bins)]))


    # Run the attack which is not using ML.
    np.random.seed(seed)
    largest_bin_results, uniform_empirical_prior_results = \
            compute_accuracy_no_ml(correlation_matrices, nbr_bins)
    if verbose:
        print('seed', seed, 'largest_bin', largest_bin_results['pred'], 
                largest_bin_results['acc'])
        print('seed', seed, 'unif emp prior', uniform_empirical_prior_results)

    return bounds, largest_bin_results, uniform_empirical_prior_results
