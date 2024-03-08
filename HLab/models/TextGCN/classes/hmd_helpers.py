import numpy as np
import scipy.sparse as sp


def normalize(adj):
    """ normalize adjacency matrix with normalization-trick that is faithful to
    the original paper.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ~D in the GCN paper
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)



def normalize_pygcn(adj):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(adj)


def moving_window_window_iterator(sentences, window):
    for sentence in sentences:
        for i in range(0, len(sentence) - window + 1):
            yield sentence[i:i + window]


def calc_pmi(X):
    Y = X.T.dot(X).astype(np.float32)
    Y_diag = Y.diagonal()
    Y.data /= Y_diag[Y.indices]
    Y.data *= X.shape[0]
    for col in range(Y.shape[1]):
        Y.data[Y.indptr[col]:Y.indptr[col + 1]] /= Y_diag[col]
    Y.data = np.maximum(0., np.log(Y.data))
    return Y
