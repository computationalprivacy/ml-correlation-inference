import argparse
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings("ignore")

from src.experiments import (experiment_correlation_extraction,
        experiment_grid_attack, 
        experiment_randomized_target_attack, 
        experiment_randomized_target_attack_model_less_only,
        experiment_randomized_target_attack_mitigations,
        experiment_real_dataset)
from src.aia import experiment_aia


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_int(val):
    """List of ints."""
    return [int(ns) for ns in val.split(',')]


def get_parser():
    parser = argparse.ArgumentParser(description=
            'Dataset correlation inference attack against ML models.')
    parser.add_argument('--experiment_name', type=str, 
            default='grid_attack')
    parser.add_argument('--save_dir', type=str, default='experiments')
    parser.add_argument('--datasets_dir', type=str, default='datasets')
    # Number of target correlation matrices to attack. Only applies to the
    # `randomized_target_attack` experiment.
    parser.add_argument('--nbr_targets', type=int, default=1000)
    parser.add_argument('--balanced_test', type=str2bool, default=False)
    parser.add_argument('--balanced_train', type=str2bool, default=False)
    parser.add_argument('--nbr_columns', type=int, default=3)
    parser.add_argument('--nbr_shadow_datasets', type=int, default=1500)
    parser.add_argument('--nbr_data_samples', type=int, default=1000)
    parser.add_argument('--target_test_size', type=float, default=0.33333)
    parser.add_argument('--shadow_test_size', type=float, default=0)
    parser.add_argument('--nbr_data_samples_bb_aux', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nbr_bins', type=int, default=3)
    # Number of divisions of the [-1, 1] interval. Used to evaluate all ranges
    # of Corr(Xi, Y) for each 1 <= i <= `nbr_columns`-1. Provided as a 
    # comma-separated list of integers, such that [-1, 1] will be divided
    # into 2*l[i] equally sized interval. The list length should be equal to 
    # `nbr_columns`-1. 
    # To be used only for --experiment_name=grid_attack.
    parser.add_argument('--lengths', type=list_int, default='10,10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nbr_cores', type=int, default=4)
    parser.add_argument('--nbr_gpus', type=int, default=2)
    # The default value of -1 means that all the figures are used, while
    # a non-negative values means that the prediction will be rounded.
    parser.add_argument('--nbr_significant_figures', type=int, default=-1)

    # Shadow model arguments.
    parser.add_argument('--shadow_model_type', type=str, default='logreg',
            help='Should be one of `logreg`, `logregdp`, `mlp`, or `mlptorch`.')
    
    # Differential privacy arguments.
    parser.add_argument('--epsilon', type=float, default=1.0,help='Value of the budget, should be a float like 1.0')

    # Meta model arguments.
    parser.add_argument('--meta_model_type', type=str, default='logreg',
            help='Should be one of `logreg`, `mlp`, or `mlptorch`.')

    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--same_seed', type=str2bool, default=False)

    # Arguments for the experiment evaluation mitigations to the black-box
    # correlation inference attack.
    parser.add_argument('--nbrs_significant_figures', type=str, 
            default='0,1,2,3,-1', 
            help='Comma-separated list of number of significant figures')
    parser.add_argument('--nbrs_data_samples_bb_aux', type=str,
            default='5,10,20,50,100,200,500,1000,2000,5000',
            help='Comma-separated list of number of queries.')

    # The correlation constraints known to the attacker when n>3. There are 
    # three possibilities:
    # 1. "column": Knowledge of the last column, i.e., the correlations between
    # Xi and Y for every i = 1, ..., n-1.
    # 2. "two": Knowledge of the correlations between X1 and Y and X2 and Y.
    # 3. "all_but_target": Knowledge of all the correlations except for the
    # target correlation between X1 and X2.
    parser.add_argument('--constraints_scenario', type=str, default='column', 
            help='Choose one between `column`, `two` and `all_but_target`')
    
    # Arguments for the real dataset evaluation.
    parser.add_argument('--dataset_name', type=str, default=
            'communities_and_crime')
    parser.add_argument('--min_repetition', type=int, default=0)
    parser.add_argument('--nbr_repetitions', type=int, default=1)
    parser.add_argument('--nbr_marginal_bins', type=int, default=100)

    # Arguments for the attribute inference attack.
    parser.add_argument('--attack_method', type=str, default='fredrikson')
    parser.add_argument('--nbr_target_records', type=int, default=100)
    parser.add_argument('--nbr_rep', type=int, default=2)
    
    return parser


def check_args(args):
    assert args.experiment_name in ['grid_attack', 
            'randomized_target_attack', 
            'randomized_target_attack_model_less_only',
            'randomized_target_attack_mitigations',
            'real_dataset_attack', 
            'aia',
            'aia2',
            'correlation_extraction',
            'dp_target_attack'], \
                    f'Invalid experiment name {args.experiment_name}'
    if args.experiment_name == 'grid_attack':
        assert len(args.lengths) == args.nbr_columns - 1
    assert args.constraints_scenario in ['column', 'two', 'all_but_target'],\
            f'ERROR: Invalid --constraints={args.constraints_scenario}.'
    assert args.dataset_name in ['communities_and_crime', 
            'communities_and_crime_v2', 'fifa19', 'fifa19_v2', 'musk'],\
            f'ERROR: Invalid --dataset_name={args.dataset_name}.'
    assert args.nbr_gpus in [1, 2], \
            f'ERROR: Invalid --nbr_gpus={args.nbr_gpus}'
    assert args.attack_method in ['cia_synth_wcai_v1', 
            'cia_synth_wcai_v2',
            'cia_synth_wcai_v3',
            'cia_synth_wcai_v4',
            'wcai', 
            'cai', 
            'fredrikson', 
            'yeom', 
            'csmia', 
            'cia_aia_model_less', 
            'cia_aia_model_based', 
            'copula_base', 
            'marginal_prior'],\
                    f'ERROR: Invalid --attack_method={args.attack_method}'



if __name__ == '__main__' :
    args = get_parser().parse_args()
    check_args(args)
    print(args)
   
    if args.experiment_name == 'randomized_target_attack':
        if args.balanced_test and args.balanced_train:
            balanced_dir = 'balanced_train_test'
        elif args.balanced_test:
            balanced_dir = 'balanced_test'
        else:
            balanced_dir = 'imbalanced'
    else:
        balanced_dir = ''
    if args.experiment_name in ['real_dataset_attack', 'aia', 
            'correlation_extraction']:
        dataset_dir = args.dataset_name
    else:
        dataset_dir = ''

    save_dir = os.path.join(args.save_dir, args.experiment_name, 
            dataset_dir, balanced_dir, f'cols-{args.nbr_columns}')
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Save directory: {save_dir}')

    if args.experiment_name == 'grid_attack':
        experiment_grid_attack(save_dir, args)
    elif args.experiment_name in ['randomized_target_attack', 
            'dp_target_attack']:
        experiment_randomized_target_attack(save_dir, args)
    elif args.experiment_name == 'randomized_target_attack_mitigations':
        experiment_randomized_target_attack_mitigations(save_dir, args)
    elif args.experiment_name == 'randomized_target_attack_model_less_only':
        experiment_randomized_target_attack_model_less_only(save_dir, args)
    elif args.experiment_name == 'real_dataset_attack':
        np.random.seed(args.seed)
        # Up to 100 repetitions, change this if you wish to run more 
        # repetitions.
        assert args.nbr_repetitions <= 100, f'ERROR: Too many repetitions, increase the number of repetitions {args.nbr_repetitions}'
        seeds = np.random.randint(10**8, size=100)
        for r in range(args.min_repetition, args.nbr_repetitions, 1):
            print(f'Executing run #{r+1} of the attack.')
            experiment_real_dataset(save_dir, args, seeds[r])
    elif args.experiment_name == 'aia':
        if not torch.cuda.is_available():
            print('CUDA is not available. Setting the device to `cpu`.')
            args.device = 'cpu'
        np.random.seed(args.seed)
        # Up to 100 repetitions, change this if you wish to run more 
        # repetitions.
        assert args.nbr_repetitions <= 100, f'ERROR: Too many repetitions, increase the number of repetitions {args.nbr_repetitions}'
        seeds = np.random.randint(10**8, size=100)
        for r in range(args.min_repetition, args.nbr_repetitions, 1):
            print(f'Executing run #{r+1} of the attack.')
            experiment_aia(save_dir, args, seeds[r])
    elif args.experiment_name == 'correlation_extraction':
        experiment_correlation_extraction(save_dir, args)
    else:
        raise ValueError(f'Unknown --experiment_name={args.experiment_name}')
