import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.helpers.utils import init_model, get_model_accuracy
from src.models import MLPTorch


def train_and_evaluate_meta_model(features, labels, model_type, use_kfold,
        other_args, verbose):
    """
    Trains the meta model to predict a label given the weights features.

    If specified, will use k-fold cross validation to get a more reliable 
    estimate of the accuracy of the meta-models.
    """
    if isinstance(features, list):
        features = np.concatenate(features)
    labels = np.array(labels)
    # If all the labels are the same, return the perfect accuracy.
    #print(features.shape, labels.shape)
    if len(set(labels)) == 1:
        return (1, 0, 1, 0), labels[0], None
    if use_kfold:
        #print(features[:2], labels[:2])
        kf = KFold(n_splits=5)
        train_accs, test_accs = [], []
        for train_index, test_index in kf.split(features):
            X_train, X_test, y_train, y_test = features[train_index], \
                    features[test_index], labels[train_index], \
                    labels[test_index]
            if len(set(y_train)) == 1:
                model = y_train[0]
            else:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                model = init_model(model_type, other_args, meta=True)
                model.fit(X_train,y_train)
            train_accs.append(get_model_accuracy(model, X_train, y_train))
            test_accs.append(get_model_accuracy(model, X_test, y_test))
        # We don't return the model here as there 5 of them.
        # TODO: Train a model on all the data and return it.
        #print(np.mean(test_accs), np.mean(train_accs))
        #print(f'test accuracy : {np.mean(test_accs)}')
        return (np.mean(test_accs), np.std(test_accs), np.mean(train_accs), \
                np.std(train_accs)), None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                test_size=0.1)
        if len(set(y_train)) == 1:
            scaler = None
            model = y_train[0]
        else:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            model = init_model(model_type, other_args, verbose=verbose, meta=True)
            model.fit(X_train, y_train)
        if model_type != 'linreg' :
            acc_train = get_model_accuracy(model, X_train, y_train)
            acc_test = get_model_accuracy(model, X_test, y_test)
        else :
            acc_train = model.score(X_train,y_train)
            acc_test = model.score(X_test,y_test)
        #print(acc_train, acc_test)
        #if isinstance(model, MLPTorch):
        #    model.to_cpu()
        #print(model_type)
        #print(f'test accuracy : {acc_test}')
        return (acc_test, 0, acc_train, 0), model, scaler


