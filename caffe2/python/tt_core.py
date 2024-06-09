## @package tt_core
# Module caffe2.python.tt_core




import numpy as np

"""
The following methods are various utility methods for using the Tensor-Train
decomposition, or TT-decomposition introduced by I. V. Oseledets (2011) in his
paper (http://epubs.siam.org/doi/abs/10.1137/090752286).

Broadly speaking, these methods are used to replace fully connected layers in
neural networks with Tensor-Train layers introduced by A. Novikov et. al. (2015)
in their paper (http://arxiv.org/abs/1509.06569). More details about each of
the methods are provided in each respective docstring.
"""


def init_tt_cores(inp_sizes, out_sizes, tt_ranks, seed=1234):
    """
    Initialize randomized orthogonalized TT-cores.

    This method should be used when a TT-layer is trained from scratch. The
    sizes of each of the cores are specified by the inp_sizes and out_sizes, and
    the respective tt_ranks will dictate the ranks of each of the cores. Note
    that a larger set of tt_ranks will result in slower computation but will
    result in more accurate approximations. The size of the ith core is:

        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].

    Note that the following relationships of lengths of each input is expected:

        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.

    Args:
        inp_sizes: list of the input dimensions of the respective cores
        out_sizes: list of the output dimensions of the respective cores
        tt_ranks: list of the ranks of the respective cores
        seed: integer to seed the random number generator

    Returns:
        cores: One-dimensional list of cores concatentated along an axis
    """
    np.random.seed(seed)

    # Assert that the sizes of each input is correct
    assert(len(inp_sizes) == len(out_sizes)), \
           "The number of input dimensions (" + str(len(inp_sizes)) + \
           ") must be equal to the number of output dimensions (" + \
           str(len(out_sizes)) + ")."

    assert(len(tt_ranks) == len(inp_sizes) + 1), \
           "The number of tt-ranks (" + str(len(tt_ranks)) + ") must be " + \
           "one more than the number of input and output dims (" + \
           str(len(out_sizes)) + ")."

    # Convert to numpy arrays
    inp_sizes = np.array(inp_sizes)
    out_sizes = np.array(out_sizes)
    tt_ranks = np.array(tt_ranks)

    # Initialize the cores array
    cores_len = np.sum(
        inp_sizes * out_sizes * tt_ranks[1:] * tt_ranks[:-1])
    cores = np.zeros(cores_len)
    cores_idx = 0
    rv = 1

    # Compute the full list of cores by computing each individual one
    for i in range(inp_sizes.shape[0]):
        shape = [tt_ranks[i],
                 inp_sizes[i],
                 out_sizes[i],
                 tt_ranks[i + 1]]

        # Precompute the shape of each core
        tall_shape = (np.prod(shape[:3]), shape[3])

        # Randomly initialize the current core using a normal distribution
        curr_core = np.dot(rv, np.random.normal(
            0, 1, size=(shape[0], np.prod(shape[1:]))))
        curr_core = curr_core.reshape(tall_shape)

        # Orthogonalize the initialized current core and append to cores list
        if i < inp_sizes.shape[0] - 1:
            curr_core, rv = np.linalg.qr(curr_core)
        cores[cores_idx:cores_idx +
              curr_core.size] = curr_core.flatten()
        cores_idx += curr_core.size

    # Normalize the list of arrays using this Glarot trick
    glarot_style = (np.prod(inp_sizes) *
                    np.prod(tt_ranks))**(1.0 / inp_sizes.shape[0])

    return (0.1 / glarot_style) * np.array(cores).astype(np.float32)


def matrix_to_tt(W, inp_sizes, out_sizes, tt_ranks):
    """
    Convert a matrix into the TT-format.

    This method will consume a 2D weight matrix such as those used in fully
    connected layers in a neural network and will compute the TT-decomposition
    of the weight matrix and return the TT-cores of the resulting computation.
    This method should be used when converting a trained, fully connected layer,
    into a TT-layer for increased speed and decreased parameter size. The size
    of the ith core is:

        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].

    Note that the following relationships of lengths of each input is expected:

        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.

    We also require that np.prod(inp_sizes) == W.shape[0] and that
    np.prod(out_sizes) == W.shape[1].

    Args:
        W: two-dimensional weight matrix numpy array representing a fully
           connected layer to be converted to TT-format; note that the weight
           matrix is transposed before decomposed because we want to emulate the
           X * W^T operation that the FC layer performs.
        inp_sizes: list of the input dimensions of the respective cores
        out_sizes: list of the output dimensions of the respective cores
        tt_ranks: list of the ranks of the respective cores

    Returns:
        new_cores: One-dimensional list of cores concatentated along an axis
   """

    # Assert that the sizes of each input is correct
    assert(len(inp_sizes) == len(out_sizes)), \
           "The number of input dimensions (" + str(len(inp_sizes)) + \
           ") must be equal to the number of output dimensions (" + \
           str(len(out_sizes)) + ")."

    assert(len(tt_ranks) == len(inp_sizes) + 1), \
           "The number of tt-ranks (" + str(len(tt_ranks)) + ") must be " + \
           "one more than the number of input and output dimensions (" + \
           str(len(out_sizes)) + ")."

    assert(W.shape[0] == np.prod(inp_sizes)), \
           "The product of the input sizes (" + str(np.prod(inp_sizes)) + \
           ") must be equal to first dimension of W (" + str(W.shape[0]) + ")."

    assert(W.shape[1] == np.prod(out_sizes)), \
           "The product of the output sizes (" + str(np.prod(out_sizes)) + \
           ") must be equal to second dimension of W (" + str(W.shape[1]) + ")."

    # W is transposed so that the multiplication X * W^T can be computed, just
    # as it is in the FC layer.
    W = W.transpose()

    # Convert to numpy arrays
    inp_sizes = np.array(inp_sizes)
    out_sizes = np.array(out_sizes)
    tt_ranks = np.array(tt_ranks)

    # Copy the original weight matrix in order to permute and reshape the weight
    # matrix. In addition, the inp_sizes and out_sizes are combined to a single
    # sizes array to use the tt_svd helper method, which only consumes a single
    # sizes array.
    W_copy = W.copy()
    total_inp_size = inp_sizes.size
    W_copy = np.reshape(W_copy, np.concatenate((inp_sizes, out_sizes)))
    order = np.repeat(np.arange(0, total_inp_size), 2) + \
            np.tile([0, total_inp_size], total_inp_size)
    W_copy = np.transpose(W_copy, axes=order)
    W_copy = np.reshape(W_copy, inp_sizes * out_sizes)

    # Use helper method to convert the W matrix copy into the preliminary
    # cores array.
    cores = tt_svd(W_copy, inp_sizes * out_sizes, tt_ranks)

    # Permute the dimensions of each of the cores to be compatible with the
    # TT-layer.
    new_cores = np.zeros(cores.shape).astype(np.float32)
    idx = 0
    for i in range(len(inp_sizes)):
        shape = (tt_ranks[i], inp_sizes[i], out_sizes[i], tt_ranks[i + 1])
        current_core = cores[idx:idx + np.prod(shape)].reshape(shape)
        current_core = current_core.transpose((1, 3, 0, 2))
        new_cores[new_cores.shape[0] - idx - np.prod(shape):
                  new_cores.shape[0] - idx] \
                  = current_core.flatten()
        idx += np.prod(shape)

    return new_cores


def tt_svd(W, sizes, tt_ranks):
    """
    Helper method for the matrix_to_tt() method performing the TT-SVD
    decomposition.

    Uses the TT-decomposition algorithm to convert a matrix to TT-format using
    multiple reduced SVD operations.

    Args:
        W: two-dimensional weight matrix representing a fully connected layer to
           be converted to TT-format preprocessed by the matrix_to_tt() method.
        sizes: list of the dimensions of each of the cores
        tt_ranks: list of the ranks of the respective cores

    Returns:
        cores: One-dimensional list of cores concatentated along an axis
   """

    assert(len(tt_ranks) == len(sizes) + 1)

    C = W.copy()
    total_size = sizes.size
    core = np.zeros(np.sum(tt_ranks[:-1] * sizes * tt_ranks[1:]),
                    dtype='float32')

    # Compute iterative reduced SVD operations and store each resulting U matrix
    # as an individual core.
    pos = 0
    for i in range(0, total_size - 1):
        shape = tt_ranks[i] * sizes[i]
        C = np.reshape(C, [shape, -1])
        U, S, V = np.linalg.svd(C, full_matrices=False)
        U = U[:, 0:tt_ranks[i + 1]]
        S = S[0:tt_ranks[i + 1]]
        V = V[0:tt_ranks[i + 1], :]

        core[pos:pos + tt_ranks[i] * sizes[i] * tt_ranks[i + 1]] = U.ravel()
        pos += tt_ranks[i] * sizes[i] * tt_ranks[i + 1]
        C = np.dot(np.diag(S), V)

    core[pos:pos + tt_ranks[total_size - 1] *
         sizes[total_size - 1] * tt_ranks[total_size]] = C.ravel()
    return core


# TODO(Surya) Write a method to convert an entire network where all fully
# connected layers are replaced by an TT layer.
def fc_net_to_tt_net(net):
    pass
