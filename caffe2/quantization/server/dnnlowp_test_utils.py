from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import numpy as np
from caffe2.python import utils, workspace
from hypothesis import assume


# This function asserts quantized results (output[1:]) are close enough to
# floating point results (output[0]).
# The error bound is derived based on assumption that there's no input
# quantization error.
def check_quantized_results_close(outputs, ref=None, symmetric=False, atol_scale=0.53):
    if ref is None:
        ref = outputs[0][0]
    if ref.size == 0:
        return
    ref_min = min(np.min(ref), 0)
    ref_max = max(np.max(ref), 0)
    if symmetric:
        ref_scale = 2 * max(abs(ref_max), abs(ref_min)) / 255
    else:
        ref_scale = (ref_max - ref_min) / 255
    # should be divided by 2 in an exact math, but divide by 1.9 here
    # considering finite precision in floating-point numbers
    atol = ref_scale * atol_scale
    for o in outputs[1:]:
        np.testing.assert_allclose(o[0], outputs[0][0], atol=atol, rtol=0)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# Make sure we won't have overflows from vpmaddubsw instruction used in fbgemm)
def avoid_vpmaddubsw_overflow_fc(
    batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max
):
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min

    # Go through the same loop again to double check we don't have any overflow
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)


# Make sure we won't have overflows from vpmaddubsw instruction used in
# fbgemm (FIXME: this assumes fbgemm is used only for NHWC and im2col
# is done in a way that input_channels is the fastest moving
# dimension).
#
# strides, pads, kernels, dilations, and sizes should be tuples with the same dimension
# (2 for 2D conv, 3 for 3D conv, and so on)
def avoid_vpmaddubsw_overflow(
    strides,
    pads,
    kernels,
    dilations,
    sizes,
    input_channels,
    output_channels,
    batch_size,
    X,
    X_min,
    X_max,
    W,
    W_min,
    W_max,
):
    ndim = len(sizes)
    dkernels = tuple((dilations[i] * (kernels[i] - 1) + 1) for i in range(ndim))
    size_cols = tuple(
        (sizes[i] + 2 * pads[i] - dkernels[i]) // strides[i] + 1 for i in range(ndim)
    )
    for out_idx in np.ndindex((batch_size,) + size_cols + (output_channels,)):
        b = out_idx[0]
        oc = out_idx[-1]
        o_spatial = out_idx[1:-1]
        for filter_idx1, filter_idx2 in pairwise(
            np.ndindex(kernels + (input_channels,))
        ):
            f0 = filter_idx1[:-1]
            ic0 = filter_idx1[-1]

            f1 = filter_idx2[:-1]
            ic1 = filter_idx2[-1]

            i0s = tuple(
                strides[i] * o_spatial[i] - pads[i] + dilations[i] * f0[i]
                for i in range(ndim)
            )
            i1s = tuple(
                strides[i] * o_spatial[i] - pads[i] + dilations[i] * f1[i]
                for i in range(ndim)
            )

            w0 = W[(oc,) + f0 + (ic0,)] - 128 - W_min
            w1 = W[(oc,) + f1 + (ic1,)] - 128 - W_min

            if all(0 <= i0s[i] < sizes[i] for i in range(ndim)):
                x0 = X[(b,) + i0s + (ic0,)] - X_min
            else:
                # padding
                x0 = -X_min

            if all(0 <= i1s[i] < sizes[i] for i in range(ndim)):
                x1 = X[(b,) + i1s + (ic1,)] - X_min
            else:
                # padding
                x1 = -X_min

            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[(oc,) + f1 + (ic1,)] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 >= (1 << 15):
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[(oc,) + f1 + (ic1,)] = int(w1_adjusted) + 128 + W_min

    # Go through the same loop again to double check we don't have any overflow
    for out_idx in np.ndindex((batch_size,) + size_cols + (output_channels,)):
        b = out_idx[0]
        oc = out_idx[-1]
        o_spatial = out_idx[1:-1]
        for filter_idx1, filter_idx2 in pairwise(
            np.ndindex(kernels + (input_channels,))
        ):
            f0 = filter_idx1[:-1]
            ic0 = filter_idx1[-1]

            f1 = filter_idx2[:-1]
            ic1 = filter_idx2[-1]

            i0s = tuple(
                strides[i] * o_spatial[i] - pads[i] + dilations[i] * f0[i]
                for i in range(ndim)
            )
            i1s = tuple(
                strides[i] * o_spatial[i] - pads[i] + dilations[i] * f1[i]
                for i in range(ndim)
            )

            w0 = W[(oc,) + f0 + (ic0,)] - 128 - W_min
            w1 = W[(oc,) + f1 + (ic1,)] - 128 - W_min

            if all(0 <= i0s[i] < sizes[i] for i in range(ndim)):
                x0 = X[(b,) + i0s + (ic0,)] - X_min
            else:
                # padding
                x0 = -X_min

            if all(0 <= i1s[i] < sizes[i] for i in range(ndim)):
                x1 = X[(b,) + i1s + (ic1,)] - X_min
            else:
                # padding
                x1 = -X_min

            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)


# strides, pads, kernels, dilations, and sizes should be tuples with the same dimension
# (2 for 2D conv, 3 for 3D conv, and so on)
def generate_convnd_inputs(
    strides,
    pads,
    kernels,
    dilations,
    sizes,
    group,
    input_channels_per_group,
    output_channels_per_group,
    batch_size,
    order,
    groupwise_quantization=False,
    preserve_activation_sparsity=False,
    preserve_weight_sparsity=False,
):
    dim = len(sizes)
    assume(all(len(a) == dim for a in [strides, pads, kernels, dilations]))
    assume(all(sizes[d] >= dilations[d] * (kernels[d] - 1) + 1 for d in range(dim)))
    input_channels = input_channels_per_group * group
    output_channels = output_channels_per_group * group
    depthwise_convolution = (
        input_channels_per_group == 1 and output_channels_per_group == 1
    )

    assert input_channels > 1
    assert output_channels > 1

    # X and W have scale 1, so exactly represented after quantization
    X_min = 0 if preserve_activation_sparsity else -77
    X_max = X_min + 255
    X_range = X_max - X_min
    if depthwise_convolution and groupwise_quantization:
        # For depthwise convolution, it's not enough to set input channel 0
        # to all X_min to avoid overflow from vpmaddubsw
        X_range /= 2
    X = np.round(
        np.random.rand(*((batch_size,) + tuple(sizes) + (input_channels,))) * X_range
        + X_min
    )
    X = X.astype(np.float32)
    if (
        batch_size != 0
        and depthwise_convolution
        and groupwise_quantization
        and not preserve_activation_sparsity
    ):
        # Put X_max in a position not to be paired with any padded value.
        # Put X_min to all positions that can be paired with the X_max value.
        #
        # This is an example of a pattern for 3x3x3
        #  .   .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .  min
        #
        #  .   .   .   .   .
        #  .   .   .   .  min
        #  .  min max min  .
        # min  .   .   .   .
        #  .   .   .   .   .
        #
        # min  .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .   .
        #  .   .   .   .   .

        # Make sure we have enough dimension
        assert X.shape[1] >= 3
        assert all(X.shape[d + 1] >= kernels[d] + 2 for d in range(1, dim))

        # Take subtensor we want to manipulate
        X_sub = X[(0,) * (X.ndim - dim - 1) + (slice(None),) * dim + (0,)]

        # Put X_max in the middle of the subtensor
        X_sub[(1,) + tuple(kernels[d] // 2 + 1 for d in range(1, dim))] = X_max

        # Put X_min to the positions that can be paired with X_max across
        # the slowest moving dimension
        X_sub[[[0, 2]] + [[kernels[d] + 1, 0] for d in range(1, dim)]] = X_min

        # Put X_min to other positions that can be paired with X_max
        for d1 in range(1, dim):
            X_sub[
                [[1]]
                + [[kernels[d2] // 2 + 1] for d2 in range(1, d1)]
                + [[kernels[d1] // 2, kernels[d1] // 2 + 2]]
                + [[kernels[d2] + 1, 0] for d2 in range(d1 + 1, dim)]
            ] = X_min
    else:
        # input channel 0 is all X_min to avoid overflow from vpmaddubsw when
        # multiplied with W_min and W_max
        X[..., 0] = X_min
        if batch_size != 0:
            X[(0,) * (X.ndim - 1) + (1,)] = X_max

    if preserve_weight_sparsity:
        W_min = -128
        W_max = 100
    else:
        W_min = -100
        W_max = W_min + 255
    W = np.round(
        np.random.rand(
            *((output_channels,) + tuple(kernels) + (input_channels_per_group,))
        )
        * (W_max - W_min)
        + W_min
    )
    W = W.astype(np.float32)
    if groupwise_quantization:
        for g in range(group):
            W[(g * output_channels_per_group,) + (0,) * (W.ndim - 1)] = W_min
            if depthwise_convolution:
                W[(g * output_channels_per_group, 1) + (0,) * (W.ndim - 2)] = W_max
            else:
                assert output_channels_per_group > 1
                W[(g * output_channels_per_group + 1,) + (0,) * (W.ndim - 1)] = W_max

            # Make sure each group has different ranges to really see the effect
            # of group-wise quantization.
            if not preserve_weight_sparsity:
                W[
                    g * output_channels_per_group : (g + 1) * output_channels_per_group,
                ] += g
    else:
        W[(0,) + (0,) * (W.ndim - 1)] = W_min
        W[(1,) + (0,) * (W.ndim - 1)] = W_max

    different_range_per_group = groupwise_quantization and not preserve_weight_sparsity
    for g in range(group):
        avoid_vpmaddubsw_overflow(
            strides,
            pads,
            kernels,
            dilations,
            sizes,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            X[..., g * input_channels_per_group : (g + 1) * input_channels_per_group],
            X_min,
            X_max,
            W[g * output_channels_per_group : (g + 1) * output_channels_per_group,],
            W_min + (g if different_range_per_group else 0),
            W_max + (g if different_range_per_group else 0),
        )

    if order == "NCHW":
        X = utils.NHWC2NCHW(X)
        W = utils.NHWC2NCHW(W)

    b = np.random.randn(output_channels).astype(np.float32)

    return X, W, b


def generate_conv_inputs(
    stride,
    pad,
    kernel,
    dilation,
    size,
    group,
    input_channels_per_group,
    output_channels_per_group,
    batch_size,
    order,
    groupwise_quantization=False,
    preserve_activation_sparsity=False,
    preserve_weight_sparsity=False,
):
    return generate_convnd_inputs(
        (stride,) * 2,
        (pad,) * 2,
        (kernel,) * 2,
        (dilation,) * 2,
        (size,) * 2,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        groupwise_quantization,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
    )


def run_conv_or_fc(
    test_case, init_net, net, X, W, b, op_type, engine, order, gc, outputs
):
    if order:
        # Conv
        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
    else:
        # FC
        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])

    # We run DNNLOWP ops multiple times to test their first runs that
    # do caching so exercises different code paths from the subsequent
    # runs

    # self.ws.run re-creates operator everytime so this test covers
    # cases when we have multiple nets sharing the same workspace
    test_case.ws.create_blob("X").feed(X, device_option=gc)
    test_case.ws.create_blob("W").feed(W, device_option=gc)
    test_case.ws.create_blob("b").feed(b, device_option=gc)
    if init_net:
        test_case.ws.run(init_net)
    for i in range(1 if engine == "" else 2):
        test_case.ws.run(net)
        Y = test_case.ws.blobs["Y"].fetch()
        if order:
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))
        else:
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine))

    # workspace.CreateNet + workspace.RunNet reuses the same operator
    if engine != "":
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)
        if init_net:
            workspace.RunNetOnce(init_net)
        workspace.CreateNet(net)
        for i in range(2):
            workspace.RunNet(net)
            Y = workspace.FetchBlob("Y")
            if order:
                outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))
            else:
                outputs.append(Output(Y=Y, op_type=op_type, engine=engine))
