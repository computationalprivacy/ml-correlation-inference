from collections import Counter
import math
import numpy as np
from scipy import stats
import diffprivlib.models as dpmodels
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import torch

from ..models import MLPTorch


def set_torch_seed(shadow_model_type, device, seed):
    if shadow_model_type == 'mlptorch':
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)


def get_device(arg_device, nbr_gpus, it):
    device = 'cpu'
    if arg_device == 'cuda':
        if nbr_gpus == 2:
            # Alternating over 2 GPUs.
            device = f'{arg_device}:{it%2}'
        else:
            device = 'cuda:0'
    return device


def get_model_weights(model):
    if isinstance(model, LogisticRegression) or isinstance(model, dpmodels.LogisticRegression):
        weights = np.concatenate((model.coef_.flatten(), model.intercept_))
    elif isinstance(model, MLPClassifier):
        weights = [layer_coefs.flatten() for layer_coefs in model.coefs_]
        weights = np.concatenate((weights, model.intercepts_))
        weights = np.concatenate((weights))
    elif isinstance(model, MLPTorch):
        weights = model.get_weights()
    else:
        raise RuntimeError(f'Invalid model type {type(model)}.')
    return weights.reshape(1, -1)


def get_weights_canonical_mlpsklearn(model):
    weights = []
    bias = []
    for layer in range(len(model.coefs_)):
        weights.append(model.coefs_[layer])
        bias.append(model.intercepts_[layer])
    canonical_weights = []
    for i in range(1, len(weights)-1):
        sort_idxs = np.argsort(np.sum(weights[i+1], axis=1))
        weights[i+1] = weights[i+1][sort_idxs]
        bias[i] = bias[i][sort_idxs]
        weights[i] = weights[i][:, sort_idxs]
        canonical_weights.append(weights[i].flatten())
        canonical_weights.append(bias[i].flatten())
    #canonical_weights.append(weights[len(weights)-1])
    #canonical_weights.append(bias[len(bias)-1])
    return np.concatenate(canonical_weights)


def get_model_weights_canonical(model):
    if isinstance(model, MLPTorch):
        return model.get_weights_canonical().reshape(1, -1)
    elif isinstance(model,MLPClassifier):
        return get_weights_canonical_mlpsklearn(model).reshape(1,-1)
    return None


def init_model(model_type, other_args=None, verbose=False, meta=False, epsilon=1.0): #added dp
    if model_type == 'logreg':
        if meta:
            return LogisticRegression(solver='lbfgs', multi_class='multinomial')
        else:
            return LogisticRegression(solver='liblinear')
        # problem of convergence when solver is lbfgs : issue solved in the 
        # latest version of sklearn. Also, liblinear solver is a good choice 
        # for small dataset, as indicated by the sklearn librairie
    elif model_type == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=100, 
                early_stopping=True, solver='adam', learning_rate_init=0.01,
                n_iter_no_change=5)
    elif model_type == 'mlptorch':
        assert len(other_args) == 7
        layer_sizes, lr, nbr_epochs, nbr_nonincreasing_epochs, weight_decay, \
                batch_size, device = other_args
        #print(nbr_nonincreasing_epochs) 
        return MLPTorch(layer_sizes, lr, nbr_epochs, nbr_nonincreasing_epochs, 
                device, weight_decay, batch_size, verbose=verbose)
    #to this point : code added for regression purposes
    elif model_type == 'linreg':
        return LinearRegression()
    elif model_type == 'decisiontree':
        return DecisionTreeClassifier(min_samples_split=50)
    elif model_type == 'logregdp': #added dp
        norm_ = stats.norm.interval(0.99, loc=0, scale=1)[1]**2
        data_norm = np.sqrt(norm_*3+1)
        return dpmodels.LogisticRegression(data_norm=data_norm, epsilon=epsilon)
    else:
        raise RuntimeError(f'Invalid model type {model_type}') 


def get_other_args(model_type, nbr_input_features, nbr_classes, device, 
        meta=False, real_dataset=False):
    if model_type == 'mlptorch':
        if meta:
            batch_size = 128
            return (nbr_input_features, 20, 10, nbr_classes), 0.001, 100, 10, 0.01, batch_size, device
            #return (nbr_input_features, 50, 20, nbr_classes), 0.001, 100, 10, 0.01, batch_size, device
        else:
            assert isinstance(real_dataset, bool), \
                    f'ERROR: `real_dataset` should be a boolean.'
            if real_dataset:
                batch_size = 128
                return (nbr_input_features, 20, 10, nbr_classes), 0.05, 100, 5, 0, batch_size, device
            else:
                batch_size = -1 # Do not batch the dataset.
                return (nbr_input_features, 20, 10, nbr_classes), 0.05, 100, 5, 0, batch_size, device
    return None


def get_model_accuracy(model, X, y):
    y_pred = predict(model, X)
    return accuracy_score(y, y_pred)


def get_confusion_matrix(model, X, y, normalize='pred'):
    if len(X) > 0:
        y_pred = predict(model, X)
        return confusion_matrix(y, y_pred, normalize=normalize)
    else:
        return np.zeros(shape=(2, 2))


def predict(model, X):
    # If the model is an integer, this means all the training labels are the
    # same. In that case, the model always predicts this label, so the accuracy
    # is equal to the proportion of samples in `y` that are equal to the label.
    if isinstance(model, np.int64):
        return [model] * len(X)
    else:
        return model.predict(X)


def predict_proba(model, X, nbr_significant_figures=-1):
    probas = model.predict_proba(X)[:, 0]
    if nbr_significant_figures >= 0:
        probas = np.round(probas, nbr_significant_figures)
    return probas


def train_model(dataset, model_type, other_args=None, verbose=False, 
        epsilon=1.0): #added dp
    X_train, y_train, X_test, y_test = dataset
    model = init_model(model_type, other_args, verbose, epsilon=epsilon)
    model = model.fit(X_train, y_train)
    #print(model.tree_.node_count)
    acc_train = get_model_accuracy(model, X_train, y_train)
    if len(X_test) > 0:
        acc_test = get_model_accuracy(model, X_test, y_test)
    else:
        acc_test = 0.0
    return model, acc_train, acc_test


def train_model_and_extract_features(dataset, model_type, other_args,
        X_attack, nbr_significant_figures, verbose=False, epsilon=1.0,
        same_seed=False): #added dp
    model, acc_train, acc_test = train_model(dataset, model_type, other_args, verbose, epsilon = epsilon) 
    if model_type in ['logreg', 'mlp', 'mlptorch', 'logregdp']: #added dp
        model_weights = get_model_weights(model)
        model_weights_canonical = get_model_weights_canonical(model)
        model_predictions = predict_proba(model, X_attack, nbr_significant_figures).reshape(1, -1)
        if model_type == 'logreg' or model_type == 'logregdp': #added dp
            combined = np.concatenate((model_weights, model_predictions), axis=1)
        elif same_seed:
            #print('I am here')
            combined = np.concatenate(
                    (model_weights, model_predictions), axis=1)
        else:
            combined = np.concatenate(
                    (model_weights_canonical, model_predictions), axis=1)
    elif model_type == 'decisiontree':
        model_weights = None
        model_weights_canonical = None
        model_predictions = predict_proba(model, X_attack, nbr_significant_figures).reshape(1, -1)
        combined = None
    else:
        raise ValueError(f'ERROR: Invalid model type {model_type}')
   
    return {'model_weights': model_weights, 
            'model_weights_canonical': model_weights_canonical,
            'model_predictions': model_predictions,
            'combined': combined}, (acc_train, acc_test)


def get_labels(correlation_matrices, i, j, nbr_bins, meta_model_type):
    """
    Extract the bins corresponding to the values of each correlation matrix at 
    row i and column j.
    """
    if meta_model_type == 'linreg' :
        labels = [ correlation_matrix[i][j] for correlation_matrix in correlation_matrices]
    else :
        labels = [ int((correlation_matrix[i][j] + 1) * nbr_bins / 2)
            for correlation_matrix in correlation_matrices]
    return np.array(labels)


def extract_correlation_matrices(datasets) :
    correlation_matrices = []
    for dataset in datasets:
        correlation_matrices.append(np.array(dataset.corr()))
    return correlation_matrices


def compute_bounds_from_constraints(c1, c2):
    theta1 = math.acos(c1)
    theta2 = math.acos(c2)
    return math.cos(theta1 + theta2), math.cos(theta1 - theta2)


def standardize_dataset(dataset, y_binary=True):
    """
    The dataset is expected to be a pandas data frame. The last column 
    (typically, the output variable y) is standardized only if `y_binary` is 
    False.
    """
    #print(f'Standardizing the input variables of the dataset.')
    standardized_dataset = dataset.copy()
    cols = dataset.columns
    for c in cols[:-1]:
        standardized_dataset[c] = (dataset[c] - dataset[c].mean()) /\
                dataset[c].std()
    if not y_binary:
        standardized_dataset[cols[-1]] = \
                ( dataset[cols[-1]] - dataset[cols[-1]].mean() ) /\
                dataset[cols[-1]].std()
    return standardized_dataset


def retrieve_matrix(predictions,bounds):
    correlation_matrix = [[0 for i in range(4)] for j in range(4)]
    for i in range(4):
        for j in range(i,4):
            if i==j:
                #diagonal
                correlation_matrix[i][j] = 1
            else :
                if j == 3 :
                    #constraints
                    correlation_matrix[i][j] = bounds[i]
                    correlation_matrix[j][i] = bounds[i]
                else:
                    correlation_matrix[i][j] = predictions[(i,j)]
                    correlation_matrix[j][i] = predictions[(i,j)]
    return correlation_matrix

def elect_representant(predictions,bounds):
    #this function will sample randomly a representant in the bin given by the output of our attack
    correlation_matrix = [[0 for i in range(4)] for j in range(4)]
    for i in range(4):
        for j in range(i,4):
            if i==j:
                #diagonal
                correlation_matrix[i][j] = 1
            else :
                if j == 3 :
                    #constraints
                    correlation_matrix[i][j] = bounds[i]
                    correlation_matrix[j][i] = bounds[i]
                else:
                    #from the formula np.random.random()*(end-start) + start with end and start equal to -1,-1/3,1/3 and 1 respectively
                    #sample randomly
                    if predictions[(i,j)] == 0:
                        representant = np.random.random()*(-1/3-(-1)) -1
                    elif predictions[(i,j)] == 1:
                        representant = np.random.random()*(1/3-(-1/3)) -1/3
                    else :
                        representant = np.random.random()*(1-1/3) + 1/3
                    correlation_matrix[i][j] = representant
                    correlation_matrix[j][i] = representant
    return correlation_matrix
