from copulas.multivariate.gaussian import GaussianMultivariate
from sklearn.model_selection import train_test_split


def generate_synthetic_dataset_using_copulas(nbr_samples, univariates,
        correlation_matrix):
    """
    Method to generate a synthetic dataset using Gaussian copulas.


    Takes as input the number of samples to generate, as well as the 
    marginal distributions and the correlation matrix.
    """
    copula = GaussianMultivariate()

    copula.covariance = correlation_matrix

    copula.columns = [f'x{i+1}' for i in range(len(univariates)-1)] + ['y']
    copula.fitted = True
    copula.univariates = univariates

    #Sample the synthetic dataset.
    synthetic_dataset = copula.sample(nbr_samples)
    return synthetic_dataset


def generate_dataset_from_correlation_matrix(correlation_matrix,
        univariates, nbr_data_samples, test_size=0):
    dataset = generate_synthetic_dataset_using_copulas(nbr_data_samples,
            univariates, correlation_matrix)
    #if recompute_correlations:
    #     correlation_matrix = dataset.corr().to_numpy()
    X = dataset.drop(columns=['y'])
    y = (dataset['y'] > 0).astype(int)
    if test_size == 0:
        return X, y, [], []
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=test_size)
    return X_train, y_train, X_test, y_test
