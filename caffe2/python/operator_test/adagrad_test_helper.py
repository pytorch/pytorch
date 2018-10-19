from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


def ref_adagrad(param_in, mom_in, grad, lr, epsilon, using_fp16=False,
                output_effective_lr=False,
                output_effective_lr_and_update=False):
    mom_in_f32 = mom_in
    param_in_f32 = param_in
    if(using_fp16):
        mom_in_f32 = mom_in.astype(np.float32)
        param_in_f32 = param_in.astype(np.float32)

    mom_out = mom_in_f32 + np.square(grad)
    effective_lr = lr / (np.sqrt(mom_out) + epsilon)
    grad_adj = effective_lr * grad
    param_out = param_in_f32 + grad_adj

    if output_effective_lr_and_update:
        if(using_fp16):
            return (param_out.astype(np.float16), mom_out.astype(np.float16),
                    effective_lr.astype(np.float16),
                    grad_adj.astype(np.float16))
        else:
            return (param_out.astype(np.float32), mom_out.astype(np.float32),
                    effective_lr.astype(np.float32),
                    grad_adj.astype(np.float32))
    elif output_effective_lr:
        if(using_fp16):
            return (param_out.astype(np.float16), mom_out.astype(np.float16),
                    effective_lr.astype(np.float16))
        else:
            return (param_out.astype(np.float32), mom_out.astype(np.float32),
                    effective_lr.astype(np.float32))

    if(using_fp16):
        return (param_out.astype(np.float16), mom_out.astype(np.float16))
    else:
        return (param_out.astype(np.float32), mom_out.astype(np.float32))


def adagrad_sparse_test_helper(parent_test, inputs, lr, epsilon,
     engine, ref_adagrad, gc, dc):
    param, momentum, grad = inputs
    momentum = np.abs(momentum)
    lr = np.array([lr], dtype=np.float32)

    # Create an indexing array containing values that are lists of indices,
    # which index into grad
    indices = np.random.choice(np.arange(grad.shape[0]),
        size=np.random.randint(grad.shape[0]), replace=False)

    # Sparsify grad
    grad = grad[indices]

    op = core.CreateOperator(
        "SparseAdagrad",
        ["param", "momentum", "indices", "grad", "lr"],
        ["param", "momentum"],
        epsilon=epsilon,
        engine=engine,
        device_option=gc)

    def ref_sparse(param, momentum, indices, grad, lr, ref_using_fp16=False):
        param_out = np.copy(param)
        momentum_out = np.copy(momentum)
        for i, index in enumerate(indices):
            param_out[index], momentum_out[index] = ref_adagrad(
                param[index],
                momentum[index],
                grad[i],
                lr,
                epsilon,
                using_fp16=ref_using_fp16
            )
        return (param_out, momentum_out)

    ref_using_fp16_values = [False]
    if dc == hu.gpu_do:
        ref_using_fp16_values.append(True)

    for ref_using_fp16 in ref_using_fp16_values:
        if(ref_using_fp16):
            print('test_sparse_adagrad with half precision embedding')
            momentum_i = momentum.astype(np.float16)
            param_i = param.astype(np.float16)
        else:
            print('test_sparse_adagrad with full precision embedding')
            momentum_i = momentum.astype(np.float32)
            param_i = param.astype(np.float32)

        parent_test.assertReferenceChecks(
            gc, op, [param_i, momentum_i, indices, grad, lr, ref_using_fp16],
            ref_sparse
        )
