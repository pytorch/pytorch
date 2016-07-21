from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def init_tt_cores(inp_sizes, out_sizes, tt_ranks, seed=1234):
    np.random.seed(seed)
    inp_sizes = np.array(inp_sizes)
    out_sizes = np.array(out_sizes)
    tt_ranks = np.array(tt_ranks)
    cores_len = np.sum(
        inp_sizes * out_sizes * tt_ranks[1:] * tt_ranks[:-1])
    cores = np.zeros(cores_len)
    cores_idx = 0
    rv = 1
    for i in range(inp_sizes.shape[0]):
        shape = [tt_ranks[i],
                 inp_sizes[i],
                 out_sizes[i],
                 tt_ranks[i + 1]]
        tall_shape = (np.prod(shape[:3]), shape[3])
        curr_core = np.dot(rv, np.random.normal(
            0, 1, size=(shape[0], np.prod(shape[1:]))))
        curr_core = curr_core.reshape(tall_shape)
        if i < inp_sizes.shape[0] - 1:
            curr_core, rv = np.linalg.qr(curr_core)
        cores[cores_idx:cores_idx +
              curr_core.size] = curr_core.flatten()
        cores_idx += curr_core.size
    glarot_style = (np.prod(inp_sizes) *
                    np.prod(tt_ranks))**(1.0 / inp_sizes.shape[0])

    return (0.1 / glarot_style) * np.array(cores).astype(np.float32)
