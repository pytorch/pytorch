import torch
import numpy as np
import scipy.sparse as sp

def foo(matrix, blocksize):
    return sp.bsr_matrix(matrix, blocksize=blocksize)


matrix = np.array(
    [[ 0. ,  0. ,  0. , -0.67160737,  0. , 0. ],
       [ 0. ,  0. , -0.88769637,  0. ,  0. , 0. ],
       [ 0.32659441,  0. ,  0. ,  0. ,  0. , 0. ],
       [ 0. ,  0. ,  0. ,  0. , -0.93967143, 0.98824336],
       [ 0.10284435,  0. ,  0. ,  0. ,  0. , 0. ],
       [ 0. ,  0. , -0.91478342,  0. ,  0. , 0. ],
       [ 0. ,  0. , -0.33557561,  0. , -1.213845  , 0. ],
       [ 0. , -1.52809994, -0.90297114,  0. ,  0. , 0. ],
       [ 0. ,  0. ,  0. ,  0. , -0.52081534, 0. ],
       [ 0. , 0. ,  0. ,  0. ,  0. , 0. ]])
blocksize = (2, 2)
torch.compile(foo)(matrix, blocksize)
