import numpy as np
import os
import pandas as pd

import src.correlations_sampler as cs
from src.synthetic_data import generate_dataset_from_correlation_matrix


def compute_distribution_over_3_classes(correlation_matrices):
    correlation_matrices = np.array(correlation_matrices)
    nbr_trials = len(correlation_matrices)
    # Bin of negative correlations.
    neg = np.sum(correlation_matrices < -1/3, axis=0) / nbr_trials
    # Bin of positive correlations.
    pos = np.sum(correlation_matrices >= 1/3, axis=0) / nbr_trials
    # Bin of small (close to zero) correlations.
    zero = 1 - neg - pos
    return neg, zero, pos


def compute_distribution_over_K_classes(correlation_matrices, K):
    """Computes the distribution over K classes for elements of the matrices."""
    correlation_matrices = np.array(correlation_matrices)
    nbr_trials = len(correlation_matrices)
    distributions = []
    bin_ends = np.linspace(-1, 1, K + 1)
    bin_ends[-1] = 1.01
    for bin_start, bin_end in zip(bin_ends, bin_ends[1:]):
        distributions.append((np.sum(correlation_matrices < bin_end, axis=0) - np.sum(correlation_matrices < bin_start, axis=0)) / nbr_trials)
    return distributions


def which_bin_is_most_covered_3(start, end):
    # Returns the majority bin in [-1, 1] for a uniform distribution over [start, end), 
    # along with the density over the majority bin.
    #assert start <= end and start >= -1 and end <= 1, 'ERROR: Invalid interval bounds'
    if start < -1/3:
        if end <= -1/3:
            return 0, 1.0
        elif end <= 1/3:
            bin1, bin2 = -1/3 - start, end - (-1/3)
            if bin1 > bin2:
                return 0, bin1 / (bin1 + bin2)
            else:
                return 1, bin2 / (bin1 + bin2)
        else:
            bin1, bin2, bin3 = -1/3 - start, 2/3, end - 1/3
            return 1, bin2 / (bin1 + bin2 + bin3)
    elif start < 1/3:
            if end <= 1/3:
                return 1, 1.0
            else:
                bin1, bin2 = 1/3 - start, end - 1/3
                if bin1 > bin2:
                    return 1, bin1 / (bin1 + bin2)
                else:
                    return 2, bin2/ (bin1 + bin2)
    else:
        return 2, 1.0
    
    
def which_bin_is_most_covered(start, end, K):
    # Returns the majority bin in [-1, 1] for a uniform distribution over [start, end), 
    # along with the density over the majority bin.
    # The number of bins is K. In [-1, 1], there are K/2 negative bins if K % 2 == 0 
    # else K-1/2 negative bins and a bin containing 0.
    assert start <= end and start >= -1 and end <= 1, 'ERROR: Invalid interval bounds'
    coverage = end-start
    bin_coverage = 2/K
    bin_start = 0
    """
    if K%2 == 0 :
        bin_start = - K/2
    else :
        bin_start = - (K-1)/2
    """
    bin_start_cdf = -1
    #bin_start_cdf : have the starting point of the bin
    #bin_start : have the start bin counter
    if 2*bin_coverage <= coverage :
        #multiple bins covered entirely : take one of them, according to the center
        center = (end+start)/2
        num=1
        while bin_start_cdf < center:
            bin_start_cdf = -1+(2*num)/K
            bin_start += 1
            num+=1
        bin_start -= 1
        bin_start_cdf -= 2/K
        #since at list on bins is covered entirely, it has to be picked for majority
        #taking the center for majority by default 
        #the majority because the probability uniform to fall into this bin, devide by the length of the new segment
        return bin_start,2/(K*(end-start))
    else :
        #no bins covered entirely : take the majority
        num=1
        while bin_start_cdf <= start:
            bin_start_cdf = -1 + (2*num)/K
            bin_start += 1
            num+=1
        if end == start :
            if start == 1 :
                return bin_start-2,1
            return (bin_start-1),1
        #bin_start_cdf above start and bin_start too (take -1 if needed)
        if bin_start_cdf < end :
            if bin_start_cdf < end - bin_coverage:
                #one bin covered entirely : take the center
                center = (end+start)/2
                num=1
                bin_start = 0
                """
                if K%2 == 0 :
                    bin_start = - K/2
                else :
                    bin_start = - (K-1)/2
                """
                while bin_start_cdf < center:
                    bin_start_cdf = -1+(2*num)/K
                    bin_start += 1
                    num+=1
                bin_start -= 1
                bin_start_cdf -= 2/K
                #since at list on bins is covered entirely, it has to be picked for majority
                #taking the center for majority by default 
                #the majority because the probability uniform to fall into this bin, devide by the length of the new segment
                return bin_start,2/(K*(end-start))
            else :
                left = bin_start_cdf - start
                right = end - bin_start_cdf
                if left > right :
                    return (bin_start-1), left/(end-start)
                else :
                    return bin_start, right/(end-start)
        else :
            return bin_start-1,1


def generate_correlation_matrices(nbr_columns, nbr_trials, bounds, lengths,
        constraints_scenario='column', balanced=False):
    """
    Generates `nbr_trials` correlation matrices, each of size `nbr_columns`.

    Each correlation matrix is over variables denoted by and ordered as 
    x_1,...,x_{n-1},y. The correlation matrix will be sampled conditionally on
    the values given by `bounds` and `lengths`, as explained below.

    If `bounds` is None, will generate purely random valid correlation matrices.

    If `lengths` is None, will generate random correlation matrices depending
    on the value of `constraints_scenario`:
        - if `constraints_scenario`=column, we assume Corr(xi, y) to be known 
        for all i=1,...,n-1 (given by Corr(xi, y)=bounds[i-1]) and we sample 
        new and valid values for the other correlations Corr(xi, xj), 
        1<=i<j<=n-1.

        - if `constraints_scenario`=two, we assume Corr(x1, y) and Corr(x2, y) 
        to be known (given by Corr(xi, y)=bounds[i-1], i=1,2) and we sample 
        new and valid values for the other correlations Corr(xi, y), 3<=i<=n-1 
        and Corr(xi, xj), 1<=i<j<=n-1.

        - if `constraints_scenario`=all_but_target, we assume the entire 
        correlation matrix to be known (and given by `bounds`), at the 
        exception of Corr(x_{n-2},x_{n-1}).

    If both `bounds` and `lengths` are not None, we generate constraints for
    the entire column such that Corr(xi, y) is uniformly sampled between 
    bounds[i] and bounds[i]+1/lenghts[i]. This can only be applied in the
    `column` scenario.

    For the meaning of `balanced`, check the method below. 
    """
    constraints, shuffle_constraints = [], True
    if bounds is None:
        # No constraints are used to generate the correlation matrices.
        constraints = [[] for _ in range(nbr_trials)]
    elif lengths is None:
        # In this case set the constraints equal to the bounds: Corr(xi, y)=
        # bounds[i-1].
        if constraints_scenario in ['column', 'two']:
            assert len(bounds.shape) == 1, \
                    'ERROR: The constraints span more than one column.'
            if constraints_scenario == 'two':
                assert len(bounds) == 2
                shuffle_constraints = False
            constraints = [bounds] * nbr_trials
        # In this case the entire matrix, except for the target correlation,
        # is used as a constraint.
        elif constraints_scenario == 'all_but_target':
            assert len(bounds.shape) == 2, \
                    'ERROR: The constraints should be given as a matrix.'
            correlation_matrices = [cs.fill_correlation_value(bounds, 
                method='no_fill') for _ in range(nbr_trials)]
            return correlation_matrices
        else:
            raise ValueError(
                f'ERROR: Invalid --constraints_scenario={constraints_scenario}')
    else:
        assert len(bounds) == len(lengths) == nbr_columns-1
        for i in range(len(bounds)):
            # The interval for Corr(xi, y) is 
            # [bounds[i], bounds[i]+1/lenghts[i]).
            constraints_xi_y = np.random.uniform(bounds[i], 
                    bounds[i]+1./lengths[i], size=nbr_trials)
            constraints.append(constraints_xi_y)
        constraints = np.array(constraints).transpose()
    correlation_matrices = [cs.generate_random_correlation_matrix_using_trig(
        nbr_columns, constraints=list(constraints[i]), 
        shuffle_constraints=shuffle_constraints,
        balanced=balanced)[0] for i in range(nbr_trials)]
    #if nbr_trials == 1:
    #    print(correlation_matrices[0])
    # Generating the majority baseline according to the most populous bin.
    return correlation_matrices


def compute_accuracy_no_ml(correlation_matrices, nbr_bins, i=0, j=1):
    """
    By default, we infer the correlation between the first two variables X1, X2.
    """
    # Generate the majority baseline according to the most populous bin.
    distribution_K = compute_distribution_over_K_classes(correlation_matrices,
            nbr_bins)
    largest_bin_results = {'acc': np.max(distribution_K, axis=0)[i][j],
            'pred': np.argmax(distribution_K, axis=0)[i][j], 
            'distribution': distribution_K}
    # Generating the majority baseline according to the uniform distribution 
    # over the empirical bounds.
    maxs = np.max(correlation_matrices, axis=0)
    mins = np.min(correlation_matrices, axis=0)
    #print('maxs', maxs, 'mins', mins)
    # Correct the leftmost and rightmost bins by setting the bin edge to -1,
    # respectively 1, if the edge is very close to -1, resp. 1.
    #if adjust:
    #    if 1 - maxs[0][1] <= 1e-2:
    #        maxs[0][1] = 1
    #    if mins[0][1] + 1 <= 1e-2:
    #        mins[0][1] = -1
    nbr_columns = len(correlation_matrices[0])
    pred, acc = which_bin_is_most_covered_bis(mins[i][j], maxs[i][j], nbr_bins)
    uniform_prior_empirical_results = {
            'acc': acc,
            'pred': pred
            }
    return largest_bin_results, uniform_prior_empirical_results


def run_attack_cell_no_ml(args):
    # The cell is specified by a list of lower bounds and interval lengths.
    # For instance if bounds=[b1, b2] and lengths=[N1, N2], the cell is equal
    # to [b1, b1+1./N1] x [b2, b2+1./N2].
    nbr_columns, nbr_trials, bounds, lengths, seed, nbr_bins = args
    np.random.seed(seed)
    correlation_matrices = generate_correlation_matrices(nbr_columns, 
            nbr_trials, bounds, lengths)
    largest_bin_results, uniform_prior_empirical_results = \
            compute_accuracy_no_ml(correlation_matrices, nbr_bins)
    return bounds, largest_bin_results, uniform_prior_empirical_results


def which_bin_is_most_covered_bis(start, end, K):
    # Returns the majority bin in [-1, 1] for a uniform distribution over 
    # [start, end), along with the density over the majority bin.
    assert start <= end and start >= -1 and end <= 1, 'ERROR: Invalid interval bounds'
    # How many interval ticks fall inside the interval [start, end] 
    # (can be equal to end).
    nbr_ticks_inside = 0
    for tick in np.linspace(-1, 1, K + 1):
        if start <= tick and tick <= end:
            nbr_ticks_inside += 1
    # We distinguish three cases.
    # (1) The interval [start, end] contains two ticks, i.e., covers at least 
    # one bin entirely. 
    # (2) The interval [start, end] contains one tick, i.e, touches two bins 
    # but none entirely.
    # (3) The interval [start, end] does not contain a tick, i.e., it is 
    # strictly included in a bin.
    if nbr_ticks_inside >= 2: # Case (1)
        # We assign the first bin entirely covered by the interval. 
        # We pay attention to edge case where start is a multiple of (2 / K),
        bin_idx = (start -  (-1) ) / (2 / K)
        if bin_idx - int(bin_idx) > 1e-8:
            bin_idx += 1
        bin_idx = int(bin_idx)
        #if there is more than one bin that is covered, we select randomly the bin predicted
        nbr_bins_inside = nbr_ticks_inside - 1
        if nbr_bins_inside > 1:
            choice = np.random.randint(0,nbr_bins_inside)
            bin_idx += choice
        return bin_idx,  2 / (K * (end - start))
    elif nbr_ticks_inside == 1: # Case (2)
        # Index and start of the bin covering the right part of the interval.
        bin_right_idx = int((end -  (-1) ) / (2 / K))
        bin_right_start = -1 + 2/K * (bin_right_idx)
        # Edge case where start == end (== tick), e.g., start=end=0 and K=2.
        if start == end:
            if end == 1:
                return bin_right_idx - 1, 1.0
            else:
                return bin_right_idx, 1.0
        # The bin covering the right part of the interval is majoritary.
        elif end - bin_right_start > bin_right_start - start:
            return bin_right_idx, (end - bin_right_start) / (end - start)
        # The bin covering the left part of the interval is majoritary.
        else:
            return bin_right_idx - 1, (bin_right_start - start) / (end - start)
    else: # Case (3)
        bin_idx = int((start -  (-1) ) / (2 / K))
        return bin_idx, 1.0

   
def compute_empirical_correlations(univariates, bounds, balanced, 
        nbr_columns=4, nbr_trials=100, nbr_data_samples=1000):
    assert len(bounds) == nbr_columns - 1
    correlation_matrices = generate_correlation_matrices(
            nbr_columns=nbr_columns,
            nbr_trials=nbr_trials,
            bounds=bounds,
            lengths=None,
            constraints_scenario='column',
            balanced=balanced)
    datasets = [generate_dataset_from_correlation_matrix(
            correlation_matrix, univariates, nbr_data_samples)
            for correlation_matrix in correlation_matrices]
    datasets = [pd.concat((X, y), axis=1) for X, y, _, _ in datasets]
    corr_xy = [dataset.corr().to_numpy()[:-1, nbr_columns-1] 
            for dataset in datasets]
    # Average value of the generated correlation Corr(Xi,Y).
    empirical_bounds = np.mean(corr_xy, axis=0)
    return empirical_bounds


def compute_shadow_bounds(bounds, univariates, balanced, max_num_iters=10, 
        max_gap=0.01, nbr_columns=4, nbr_trials=100, nbr_data_samples=1000):
    """
    Compute shadow bounds rho'(Xi,Y) such that datasets generated from the 
    one-way marginals under these constraints have empirical correlations close 
    to rho(Xi,Y) (given by `bounds`).
    
    bounds: A list of correlations rho(Xi,Y) of size `nbr_columns`-1.
    univariates: One-way marginals of the dataset.
    balanced: Whether the shadow matrices used to estimate the empirical 
        correlations should be generated uniformly in their interval.
    """
    num_iters = 0
    shadow_bounds = bounds.copy()

    gap = [0] * (nbr_columns - 1)
    while num_iters < max_num_iters:
        #print('num iters', num_iters, 'shadow_bounds', shadow_bounds)
        empirical_bounds = compute_empirical_correlations(univariates,
                shadow_bounds, balanced, nbr_columns, nbr_trials, 
                nbr_data_samples) 
        #print('empirical bounds', empirical_bounds)
        # Are the bounds calculated on the generated data close to the target 
        # bounds?
        empirical_gap = np.max(np.abs(empirical_bounds - bounds))
        if empirical_gap < max_gap:
            break
        # If not, attempt to reduce the gap.
        for bi in range(len(bounds)):
            gap[bi] = bounds[bi] - empirical_bounds[bi]
            # Adjust the constraint using a simple linear update rule.
            shadow_bounds[bi] = np.maximum(-0.999999, 
                    np.minimum(0.999999, shadow_bounds[bi] + gap[bi]/2)) 
        
        num_iters += 1

    return shadow_bounds
