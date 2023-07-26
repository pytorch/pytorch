from functools import partial

import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core


def ref_adagrad(
    param_in,
    mom_in,
    grad,
    lr,
    epsilon,
    using_fp16=False,
    output_effective_lr=False,
    output_effective_lr_and_update=False,
    decay=1.0,
    row_wise=False,
    weight_decay=0.0,
    counter_halflife=-1,
    count=None,  # only used when counter_halflife != -1
):
    mom_in_f32 = mom_in
    param_in_f32 = param_in
    if using_fp16:
        mom_in_f32 = mom_in.astype(np.float32)
        param_in_f32 = param_in.astype(np.float32)

    if count and count > 0 and counter_halflife > 0:
        weight_decay *= counter_halflife / count
    grad_temp = grad + weight_decay * param_in_f32
    if row_wise:
        mom_out = decay * mom_in_f32 + np.mean(np.square(grad_temp))
    else:
        mom_out = decay * mom_in_f32 + np.square(grad_temp)
    effective_lr = lr / (np.sqrt(mom_out) + epsilon)
    grad_adj = effective_lr * grad_temp
    param_out = param_in_f32 + grad_adj

    if output_effective_lr_and_update:
        if using_fp16:
            return (
                param_out.astype(np.float16),
                mom_out.astype(np.float16),
                effective_lr.astype(np.float16),
                grad_adj.astype(np.float16),
            )
        else:
            return (
                param_out.astype(np.float32),
                mom_out.astype(np.float32),
                effective_lr.astype(np.float32),
                grad_adj.astype(np.float32),
            )
    elif output_effective_lr:
        if using_fp16:
            return (
                param_out.astype(np.float16),
                mom_out.astype(np.float16),
                effective_lr.astype(np.float16),
            )
        else:
            return (
                param_out.astype(np.float32),
                mom_out.astype(np.float32),
                effective_lr.astype(np.float32),
            )

    if using_fp16:
        return (param_out.astype(np.float16), mom_out.astype(np.float16))
    else:
        return (param_out.astype(np.float32), mom_out.astype(np.float32))


def adagrad_sparse_test_helper(
    parent_test,
    inputs,
    lr,
    epsilon,
    engine,
    ref_adagrad,
    gc,
    dc,
    row_wise=False,
    weight_decay=0.0,
    counter_halflife=-1,
):
    param, momentum, grad = inputs
    if row_wise:
        # For row-wise adagrad, only take the first element of each row
        momentum = momentum.reshape(momentum.shape[0], -1)[:, 0]
    momentum = np.abs(momentum)
    lr = np.array([lr], dtype=np.float32)
    count = None
    if counter_halflife != -1:
        count = np.random.rand(param.shape[0])

    # Create an indexing array containing values that are lists of indices,
    # which index into grad
    if grad.size == 0:
        indices = np.empty(shape=(0,), dtype=int)
    else:
        indices = np.random.choice(
            np.arange(grad.shape[0]),
            size=np.random.randint(grad.shape[0]),
            replace=False,
        )

    # Sparsify grad
    grad = grad[indices]

    op = core.CreateOperator(
        "RowWiseSparseAdagrad" if row_wise else "SparseAdagrad",
        ["param", "momentum", "indices", "grad", "lr"] if count is None else ["param", "momentum", "indices", "grad", "lr", "count"],
        ["param", "momentum"],
        epsilon=epsilon,
        weight_decay=weight_decay,
        counter_halflife=counter_halflife,
        engine=engine,
        device_option=gc,
    )

    def ref_sparse(param, momentum, indices, grad, lr, count=None, ref_using_fp16=False):
        param_out = np.copy(param)
        momentum_out = np.copy(momentum)
        # Need to do this because it's possible ref_adagrad's using_fp16 could
        # have been already specialized.
        ref_adagrad_temp = (
            partial(ref_adagrad, using_fp16=ref_using_fp16)
            if ref_using_fp16
            else ref_adagrad
        )
        for i, index in enumerate(indices):
            param_out[index], momentum_out[index] = ref_adagrad_temp(
                param[index],
                momentum[index],
                grad[i],
                lr,
                epsilon,
                weight_decay=weight_decay,
                counter_halflife=counter_halflife,
                count=None if count is None else count[index],
            )
        return (param_out, momentum_out)

    ref_using_fp16_values = [False]
    if gc == hu.gpu_do and not row_wise:
        ref_using_fp16_values.append(True)

    for ref_using_fp16 in ref_using_fp16_values:
        if ref_using_fp16:
            print("test_sparse_adagrad with half precision embedding")
            momentum_i = momentum.astype(np.float16)
            param_i = param.astype(np.float16)
        else:
            print("test_sparse_adagrad with full precision embedding")
            momentum_i = momentum.astype(np.float32)
            param_i = param.astype(np.float32)

        parent_test.assertReferenceChecks(
            gc,
            op,
            [param_i, momentum_i, indices, grad, lr, count, ref_using_fp16],
            ref_sparse
        )
