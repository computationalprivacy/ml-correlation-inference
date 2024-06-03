import numpy as np
import cvxpy as cp
import time


def check_positive_definite(x):
    return np.all(np.linalg.eigvals(x) > 0)


def check_positive_semidefinite(x):
        return np.all(np.linalg.eigvals(x) >= 0)


def project_matrix_to_psd(matrix, constraint_indexes=None, verbose=False):
    """
    This method projects a given matrix into the space of positive 
    semi-definite (PSD) matrices.

    The goal is to find the closest PSD matrix in terms of the Frobenius
    norm. We write this as a constrained optimization program:

    min ||proj - matrix||_F
    subject to:
        1. proj is a PSD matrix
        2. all elements of proj are valid correlations, i.e., between -1 and 1.
        3. the diagonal elements of proj are equal to 1 (i.e., a variable's
        corelation to itself is 1).

    If constraint_indexes is not None, then we add new constraints:
        4. the correlation for the pairs should be identical to those in the 
        input matrix.
    
    
    Arguments 
    -------------
    
    matrix : 2-dim array of floats.
        Matrix to be projected. 
    constraint_indexes: set of (int, int) pairs.
        The coordinates in the matrix for which we set equality constraints.
    verbose: bool
        Whether to print debugging info.
        
    Result
    -------------
    
    projection : 2-dim array of floats of same shape as `matrix`.
        The PSD matrix that is closest to `matrix` using the Frobenius norm. 
        
    Example
    -------------
    
    >>>project_matrix_to_psd([[1,0.5,0.2],[0.1,1,0.6],[-0.5,0.1,1]])
    [[ 1.    0.3  -0.15]
     [ 0.3   1.    0.35]
     [-0.15  0.35  1.  ]])
    """
    assert len(matrix) == len(matrix[0]), 'ERROR: the matrix is not square.'
    size = len(matrix)
    projection = cp.Variable((size, size), PSD=True)
    # Constraint 2 and 3: the matrix is a valid correlation matrix.
    constraints = [projection <= 1, projection >= -1, cp.diag(projection) == 1]
    # Constraint 4 regarding which elements should be the same as in the original
    # matrix.
    if constraint_indexes is not None:
        constraints += [projection[i-1][j-1] == matrix[i-1][j-1] 
                for (i, j) in constraint_indexes]

    prob = cp.Problem(cp.Minimize(cp.sum_squares(projection - matrix)), 
            constraints)
    result = prob.solve(verbose=verbose, eps=1e-10)
    return projection.value, result, prob.solver_stats.solve_time


def generate_random_correlation_matrix_using_optim(nbr_columns, 
        constraints, initialization='sym_diag_one'):
    """Generates a random correlation matrix. 
    
    Works by generating a symmetric matrix with elements sampled uniformly 
    between -1 and 1. If specificed, the diagonal can be set to 1.

    Additionally, elements at a subset of indexes (i, j) can be set to 
    specific values v = constraints[(i, j)].

    Then, the matrix is projected on the PSD space.

    Returns the projection matrix.
    """
    # Initialize a matrix with elements uniformly sampled between -1 and 1.
    matrix = np.random.uniform(-1, 1, size=(nbr_columns, nbr_columns))
    if initialization == 'sym_diag_one':
        # Enforce that the matrix is symmetric.
        triu = np.triu(matrix, k=1)
        # Additionally, set the diagonal to 1.
        matrix = triu + triu.T + np.eye(nbr_columns)
    elif initialization == 'sym':
        # Enforce that the matrix is symmetric.
        matrix = np.triu(matrix, k=0) + np.triu(matrix, k=1).T
    else:
        raise ValueError('Unimplemented initialization method.')
    if constraints is not None:
        for (i, j) in constraints:
            matrix[i-1][j-1] = constraints[(i,j)]
            matrix[j-1][i-1] = constraints[(i,j)]
    constraint_indexes = sorted(list(constraints.keys()))
    projection, distance, elapsed_time = project_matrix_to_psd(matrix,
            constraint_indexes)
    #projection = projection.tolist()
    return projection, matrix, distance, elapsed_time


def generate_random_correlation_matrix_using_trig(nbr_columns, constraints=[], 
        K=0.01, shuffle_constraints=True, balanced=False):
    """"
    A method to generate randomized correlation matrices with equality 
    constraints for the last column.
    
    By default, the method generates a random correlation matrix using the 
    algorithm by Numpacharoen and Atsawarungruangkit (2012) 
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0048902.
    
    If contraints are given, they will be enforced on the last column of the 
    correlation matrix using an adaptation of the above algorithm.
    
    Args:
        nbr_columns: The number of columns of the correlation matrix.
        constraints: (default = []) List of floats. The i-th element denotes 
            the value of the output correlation matrix at position 
            (i, nbr_columns). Expects either an empty list (no constraints)
            or a list of length nbr_columns-1.
        K: (default=0.01) Threshold factor of stability ensuring that the 
            algorithm works properly (cf paper).
        
    Returns:
        A random correlation matrix and the runtime.
    """
    start_time = time.time()
    # Step 0: When there are constraints, shuffle them to avoid a bias given 
    # by the variable order used to generate the correlations. 
    if len(constraints) > 0:
        # Double-check the constraints. If the list of constraints is not 
        # empty, ensure it contains a valid constraint for each cell.
        assert len(constraints) == nbr_columns - 1 or \
                (len(constraints) == 2 and shuffle_constraints==False)
        for c in constraints:
            assert -1 <= c and c <= 1
        if shuffle_constraints:
            if balanced and nbr_columns > 3:
                shuffled_idxs = np.random.permutation(
                        np.arange(2, nbr_columns-1, 1))
                shuffled_idxs = [0, 1] + list(shuffled_idxs)
            else:
                # Shuffle the constraints using a random permutation.
                shuffled_idxs = np.random.permutation(nbr_columns-1)
            shuffled_constraints = [constraints[i] for i in shuffled_idxs]
        else:
            shuffled_idxs = np.arange(nbr_columns-1)
            shuffled_constraints = constraints
        # Inverse permutation to be used at the very end.
        inverse_idxs = list(np.argsort(shuffled_idxs)) + [nbr_columns - 1]
    # Step 1: Initialize the correlation matrix C and the helper matrix B.
    C = np.zeros((nbr_columns, nbr_columns))
    for i in range(1, nbr_columns):
        # When there are constraints, we reorder the variables X1, ..., Xn to
        # Xn, ..., X1. We do this because we want the last column to be fixed,
        # while the algorithm only allows to fix the first column.
        if i <= len(constraints):
            C[i][0] = shuffled_constraints[i-1]
        else:
            # Otherwise the first column is initialized randomly.
            C[i][0] = 2*np.random.random() - 1
    C = fill_correlation_matrix(C, K)
    # Step 3: If there are no constraints, simply permute the variables to 
    # to ensure that there is no bias. 
    # If there are constraints, the columns need to be reordered to make sure 
    # each constraint appears in the desired position. This is done by invertin
    # the two orderings in the *inverse* order that they were applied.
    if len(constraints) == 0:
        if balanced:
            shuffled_columns = np.random.permutation(np.arange(2, nbr_columns, 1))
            order = [0, 1] + list(shuffled_columns)
        else:
            order = np.random.permutation(nbr_columns)
    elif len(constraints) == 2 and nbr_columns > 3:
        shuffled_columns = np.random.permutation(np.arange(3, nbr_columns, 1))
        order = [1, 2] + list(shuffled_columns) + [0]
        #print(order)
    else:
        # Move the first column (corresponding to Y) to the first position.
        order = list(np.arange(1, nbr_columns, 1)) + [0]
    C = C[order][:, order]
    
    # Step 4: If there are constraints, revert the ordering between the 
    # constraints.
    if len(constraints) > 0:
        C = C[inverse_idxs][:, inverse_idxs]
    return C, time.time() - start_time


def fill_correlation_matrix(C, K):
    nbr_columns = len(C)
    B = np.tril(np.ones(nbr_columns), 0)
    for i in range(1, nbr_columns):
        B[i][0] = C[i][0]
        for j in range(1, i+1):
            B[i][j] = np.sqrt(1-C[i][0]**2)
    # Step 2 : Fill the values of the correlation matrix row by row and then 
    # within each row, from left to right.
    for i in range(2, nbr_columns):
        for j in range(1, i):
            B1 = np.matmul(B[j][0:j], np.transpose(B[i][0:j]))
            B2 = B[j][j] * B[i][j]
            #Z, Y : upper/lower bound respectively
            Z = B1 + B2
            Y = B1 - B2
            if B2 < K:
                C[i][j] = B1
                cosinv = 0
            else :
                C[i][j] = Y + (Z-Y) * np.random.random()
            cosinv = (C[i][j] - B1) / B2
            if cosinv > 1:
                for k in range(j+1, nbr_columns):
                    B[i][k] = 0
            elif cosinv < -1:
                B[i][j] = - B[i][j]
                for k in range(j+1, nbr_columns):
                    B[i][k] = 0
            else :
                B[i][j] = B[i][j] * cosinv
                sinTheta = np.sqrt(1-cosinv**2)
                for k in range(j+1, nbr_columns):
                    B[i][k] = B[i][k] * sinTheta
    C = C + np.transpose(C) + np.eye(nbr_columns)
    return C


def fill_correlation_value(C, K=0.01, method='no_fill'):
    """
    Given a nxn matrix C of correlation constraints, fill in the missing 
    value in position (1,2) uniformly at random within its boundaries.
    
    The boundaries are determined from the other values of the matrix using
    the spherical parametrization of correlation matrices. 
    
    The method relies on the full/partial Choleski decomposition of C. To 
    compute the full decomposition of C, the value in position (1,2) should be
    such that C is a valid correlation matrix. The attacker does however not 
    know the true value C[1][2]. We thus support several methods:

    1. method=fill_true_value: This method uses the true value and should only
    be used for testing.
    2. method=fill_random_value: A value is sampled uniformly at random within
    the boundaries. Since we use rejection sampling, the method can take longer
    when the number of columns is large.
    3. method=no_fill: No placeholder value is used. Instead, we compute a 
    partial Choleski decomposition of C (on the submatrix consisting of the first
    n-1 rows and columns of C). We then compute the remaining values one by one.
    """
    C = np.copy(C)
    n = len(C)
    for i in range(n):
        assert C[i][i] == 1
    # Re-order the columns as Y, Xn-1, ..., X1.
    order = [n - 1] + list(np.arange(n-1)[::-1])
    C = C[order][:, order]
    
    if method == 'fill_true_value':
        # The original value passed as argument trivially results in a valid correlation
        # matrix. The attack cannot use the original value as this is what it's
        # trying to predict. We implement this method in order to test the 
        # correctness of the other methods.
        assert check_positive_definite(C)
        #print('Reorderd C', C)
        B = np.linalg.cholesky(C)
    elif method == 'fill_random_value':
        # First, we drop the true value C[1,2] by sampling a new value such
        # that C remains positive definite. Note that this is likely to take
        # longer for large number of variables n.
        while True:
            u = np.random.random()*2 -1
            C[n-2][n-1] = C[n-1][n-2] = u
            if check_positive_definite(C):
                break
        assert check_positive_definite(C)
        #print('Reorderd C', C)
        B = np.linalg.cholesky(C)
    elif method == 'no_fill':
        # We don't attempt to initialize the missing correlation value.
        # We will just sample it within its boundaries.
        # To do this, we first compute the first n-1 rows and columns of B.
        Cprim = C[:-1][:, :-1]
        assert check_positive_definite(Cprim)
        B = np.ones((n,n))
        B[:-1][:, :-1] = np.linalg.cholesky(Cprim)
        # Next, we fill in the last row of B.
        B[n-1][0] = C[n-1][0]
        for i in range(1, n-2):
            B[n-1][i] = C[n-1][i] - np.matmul(B[i][:i], B[n-1][:i].transpose())
            # TODO: Deal with the edge case where B[i][i] = 0. Note that this is
            # unlikely to occur when C is random as is the case in our experiments.
            B[n-1][i] /= B[i][i]
    else:
        raise ValueError(f'ERROR: Invalid method {method}')
             
    # Compute sqrt(1 - B_{n,1}**2 - B_{n,2}**2 - ... - B_{n,n-2}**2).
    # This allows to fill in B_{n,n-1}.
    B[n-1][n-2] = np.sqrt(1-np.sum(B[n-1][:(n-2)]**2))
    
    B1 = np.matmul(B[n-2][0:(n-2)], np.transpose(B[n-1][0:(n-2)]))
    B2 = B[n-2][n-2] * B[n-1][n-2]
    #Z, Y : upper/lower bound respectively
    Z = B1 + B2
    Y = B1 - B2
    if B2 < K:
        C[n-1][n-2] = B1
    else:
        C[n-1][n-2] = Y + (Z-Y) * np.random.random()
        
    C[n-2][n-1] = C[n-1][n-2]
    
    #print('Matrix C', C)
    # Order the columns back as X1, ..., Xn-1, Y.
    C = C[order][:, order]
    return C


def generate_random_correlation_matrix(nbr_columns, constraints=dict(), 
        method='optim', initialization_optim='sym_diag_one',K=0.01):
    """
    Generate a random correlation matrix using the specified methods.

    nbr_columns: int. Number of columns
    constraints: dict of (int, int) -> int. default: dict()
        The keys are coordinates in the matrix and the values are the 
        correlations for which we want to enforce equality constraints.

    Returns a random correlation matrix, together with the number of trials it
    took to obtain a positive definite matrix.
    """
    nbr_trials = 1
    while True:
        if method == 'optim':
            correlation_matrix, _, _, _ = \
                    generate_random_correlation_matrix_using_optim(nbr_columns,
                            constraints, initialization_optim)
        elif method == 'trigonometric':
            correlation_matrix, _ = \
                    generate_random_correlation_matrix_using_trig(
                            nbr_columns, constraints, K)
        else:
            raise ValueError('ERROR: Unimplemented method.')
        if check_positive_definite(correlation_matrix):
            break
        else:
            nbr_trials += 1
    return correlation_matrix, nbr_trials
