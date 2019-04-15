from __future__ import absolute_import, division, print_function, unicode_literals

import copy
from collections import defaultdict

import numpy as np
from caffe2.python import core, utils
from caffe2.python.fb import hardcode_scale_zp


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def blob_uses(net, blob):
    u = []
    for i, op in enumerate(net.op):
        if blob in op.input or blob in op.control_input:
            u.append(i)
    return u


def fuse_first_bn(net, params, removed_tensors):
    net = copy.deepcopy(net)
    params = copy.deepcopy(params)

    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue

        if current.type not in ("Conv", "ConvTranspose") or next_.type != "SpatialBN":
            continue
        if (
            len(blob_uses(net, current.output[0])) != 1
            and current.output[0] != next_.output[0]
        ):
            # Can't fuse if more than one user unless SpatialBN is inplace
            continue

        # else, can fuse
        conv = current
        bn = next_
        fused_conv = copy.deepcopy(conv)
        fused_conv.output[0] = bn.output[0]
        conv_weight = params[conv.input[1]]
        if len(conv.input) > 2:
            conv_bias = params[conv.input[2]]
        else:
            conv_bias = np.zeros(len(params[bn.input[2]])).astype(np.float32)

        bn_scale = params[bn.input[1]]
        bn_bias = params[bn.input[2]]
        bn_running_mean = params[bn.input[3]]
        bn_running_var = params[bn.input[4]]

        # First, BN computation can be phrased as follows:
        # (X - running_mean) * (1.0 / sqrt(running_var + eps)) *
        # bn_scale + bias
        # Thus, we can rewrite bn_scale as:
        # X * bn_scale * 1.0 / (sqrt(running_var + eps)) + (bias -
        # running_mean * (1.0 / sqrt(running_var + eps)) * bn_scale)
        # Thus, can just have the affine transform
        # X * A + B
        # where
        # A = bn_scale * 1.0 / (sqrt(running_var + eps))
        # B =  (bias - running_mean * (1.0 / sqrt(running_var + eps))
        # * bn_scale)
        eps = 1.0e-5
        for arg in bn.arg:
            if arg.name == "epsilon":
                eps = arg.f
        A = bn_scale * 1.0 / (np.sqrt(bn_running_var + eps))
        B = bn_bias - bn_running_mean * A

        # This identity should hold if we have correctly fused
        # np.testing.assert_array_equal(
        #     params[conv.output[0]] * A + B,
        #     params[bn.output[0]])

        # Now, we have that the computation made is the following:
        # ((X `conv` W) + b) * A + B
        # Then, we can simply fuse this as follows:
        # (X `conv` (W * A)) + b * A + B
        # which is simply
        # (X `conv` Q) + C
        # where

        # Q = W * A
        # C = b * A + B

        # For ConvTranspose, from the view of convolutions as a
        # Toepeliz multiplication, we have W_ = W^T, so the weights
        # are laid out as (R, S, K, K) (vs (S, R, K, K) for a Conv),
        # so the weights broadcast slightly differently. Remember, our
        # BN scale 'B' is of size (S,)

        A_ = (
            A.reshape((-1,) + tuple([1] * (conv_weight.ndim - 1)))
            if conv.type == "Conv"
            else A.reshape((1, -1) + tuple([1] * (conv_weight.ndim - 2)))
        )

        C = conv_bias * A + B
        Q = conv_weight * A_

        assert params[conv.input[1]].shape == Q.shape
        if len(conv.input) > 2:
            assert params[conv.input[2]].shape == C.shape
        else:
            assert bn_bias.shape == C.shape

        params[conv.input[1]] = Q
        if len(conv.input) > 2:
            params[conv.input[2]] = C
        else:
            params[bn.input[2]] = C
            fused_conv.input.append(bn.input[2])

        new_ops = net.op[:i] + [fused_conv] + net.op[j + 1 :]
        del net.op[:]
        removed_tensors.append(bn.input[1])
        if len(conv.input) > 2:
            removed_tensors.append(bn.input[2])
        removed_tensors.append(bn.input[3])
        removed_tensors.append(bn.input[4])
        del params[bn.input[1]]
        if len(conv.input) > 2:
            del params[bn.input[2]]
        del params[bn.input[3]]
        del params[bn.input[4]]
        net.op.extend(new_ops)
        break
    return net, params, removed_tensors


def fuse_bn(net, params, ignore_failure):
    # Run until we hit a fixed point
    removed_tensors = []
    while True:
        (next_net, next_params, removed_tensors) = fuse_first_bn(
            net, params, removed_tensors
        )
        if len(next_net.op) == len(net.op):
            if any(op.type == "SpatialBN" for op in next_net.op) and not ignore_failure:
                raise Exception(
                    "Model contains SpatialBN op after fusion: %s", next_net
                )
            return (next_net, next_params, removed_tensors)
        net, params, removed_tensors = (next_net, next_params, removed_tensors)


def fuse_first_scale(net, params, removed_tensors):
    net = copy.deepcopy(net)
    params = copy.deepcopy(params)

    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue

        if (
            current.type != "SpatialBN"
            or next_.type != "Mul"
            or len(net.op) <= j + 1
            or net.op[j + 1].type != "Add"
        ):
            continue

        # else, can fuse
        bn = current
        mul = next_
        add = net.op[j + 1]

        fused_bn = copy.deepcopy(bn)
        fused_bn.output[0] = add.output[0]
        bn_scale = params[bn.input[1]]
        mul_scale = params[mul.input[1]]
        bn_bias = params[bn.input[2]]
        add_bias = params[add.input[1]]

        params[bn.input[1]] = bn_scale * mul_scale
        params[bn.input[2]] = mul_scale * bn_bias + add_bias

        new_ops = net.op[:i] + [fused_bn] + net.op[j + 2 :]
        del net.op[:]
        removed_tensors.append(mul.input[1])
        removed_tensors.append(add.input[1])
        del params[mul.input[1]]
        del params[add.input[1]]
        net.op.extend(new_ops)
        break
    return net, params, removed_tensors


def fuse_scale(net, params, ignore_failure):
    # Run until we hit a fixed point
    removed_tensors = []
    while True:
        (next_net, next_params, removed_tensors) = fuse_first_scale(
            net, params, removed_tensors
        )
        if len(next_net.op) == len(net.op):
            return (next_net, next_params, removed_tensors)
        net, params, removed_tensors = (next_net, next_params, removed_tensors)


def fuse_first_relu(net, ignore_op_with_output=None):
    net = copy.deepcopy(net)

    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue

        if current.type not in ("Conv", "Sum") or next_.type != "Relu":
            continue

        if ignore_op_with_output and current.output[0] in ignore_op_with_output:
            continue

        # else, can fuse
        conv = current
        relu = next_
        fused_conv = copy.deepcopy(conv)
        fused_conv.type = "ConvRelu" if current.type == "Conv" else "SumRelu"
        fused_conv.output[0] = relu.output[0]

        new_ops = net.op[:i] + [fused_conv] + net.op[j + 1 :]
        del net.op[:]
        net.op.extend(new_ops)
        break
    return net


def fuse_relu(net, ignore_failure, ignore_op_with_output=None):
    # Run until we hit a fixed point
    while True:
        next_net = fuse_first_relu(net, ignore_op_with_output)
        if len(next_net.op) == len(net.op):
            if any(op.type == "Relu" for op in next_net.op) and not ignore_failure:
                raise Exception("Model contains Relu op after fusion: %s", next_net)
            return next_net
        net = next_net


def last_producer(ops, blob):
    for (i, op) in reversed(list(enumerate(ops))):
        if op.output[0] == blob:
            return i
    raise ValueError("Failed to find last producer of blob, %s", blob)


def swap_first_concat_relu(net, ignore_op_with_output=None):
    net = copy.deepcopy(net)

    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue

        if current.type != "Concat" or next_.type != "Relu":
            continue

        if ignore_op_with_output and current.output[0] in ignore_op_with_output:
            continue

        # else, can swap
        concat = copy.deepcopy(current)
        relu = copy.deepcopy(next_)
        pre_ops = copy.deepcopy(net.op[:i])
        post_ops = copy.deepcopy(net.op[j + 1 :])

        # Delete the Relu after Concat
        concat.output[0] = relu.output[0]

        # Insert Relu after each op that produces inputs to Concat
        for blob in concat.input:
            k = last_producer(pre_ops, blob)
            producer = pre_ops[k]
            assert producer.output[0] == blob
            producer.output[0] = blob + "_pre_relu"

            new_relu = copy.deepcopy(relu)
            new_relu.input[0] = producer.output[0]
            new_relu.output[0] = blob

            pre_ops = pre_ops[: k + 1] + [new_relu] + pre_ops[k + 1 :]

        new_ops = pre_ops + [concat] + post_ops
        del net.op[:]
        net.op.extend(new_ops)
        break
    return net


def swap_concat_relu(net, ignore_op_with_output=None):
    # Run until we hit a fixed point
    while True:
        next_net = swap_first_concat_relu(net, ignore_op_with_output)
        if len(next_net.op) == len(net.op):
            return next_net
        net = next_net


def add_version_to_conv_bias(net, init_net):
    """
    In architectures such as FPN (https://arxiv.org/abs/1612.03144), few Conv
    ops share the same weight and bias and are run at different scales of
    the input. Since 'bias_scale = input_scale * weight_scale', sharing the
    same bias blob among multiple Conv ops means that we need different bias
    scale for each of the ops. To achieve this, we just duplicate those bias
    blobs that are used by multiple Conv ops before performing int8 rewrite.
    """
    bias_count = defaultdict(int)
    for op in net._net.op:
        if "Conv" in op.type and len(op.input) >= 3:
            bias_count[op.input[2]] += 1

    bias_fill_op = {}
    for op in init_net._net.op:
        if bias_count[op.output[0]] > 1:
            bias_fill_op[op.output[0]] = op

    bias_version = defaultdict(int)
    for op in net._net.op:
        if "Conv" in op.type and len(op.input) >= 3:
            bias = op.input[2]
            if bias_count[bias] <= 1:
                continue

            version = bias_version[bias]
            bias_version[bias] += 1
            if version == 0:
                continue

            new_bias = bias + "_v" + str(version)
            fill_op = copy.deepcopy(bias_fill_op[bias])
            fill_op.output[0] = new_bias
            init_net._net.op.extend([fill_op])
            op.input[2] = new_bias
            net._net.external_input.append(new_bias)


def add_quantization_param_args_(op, q_param):
    op.arg.extend(
        [
            utils.MakeArgument("Y_scale", q_param.scale),
            utils.MakeArgument("Y_zero_point", q_param.zero_point),
        ]
    )


def choose_quantization_params(tensor_min, tensor_max, preserve_sparsity=False):
    if tensor_min < 0 and tensor_max > 0 and preserve_sparsity:
        symmetric_qmin = -(255 // 2 + 1)
        symmetric_qmax = 255 // 2
        max_scale = max(
            abs(tensor_min / symmetric_qmin), abs(tensor_max / symmetric_qmax)
        )
        tensor_min = max_scale * symmetric_qmin
        tensor_max = max_scale * symmetric_qmax

    q_param = hardcode_scale_zp.choose_quantization_params(tensor_min, tensor_max)

    if tensor_min < 0 and tensor_max > 0 and preserve_sparsity:
        q_param = hardcode_scale_zp.QuantizationParam(q_param.scale, 128)

    return q_param


def add_quantization_param_args(op, tensor, preserve_sparsity=False):
    tensor_min = 0 if tensor.size == 0 else tensor.min()
    tensor_max = 0 if tensor.size == 0 else tensor.max()

    q_param = choose_quantization_params(tensor_min, tensor_max, preserve_sparsity)

    add_quantization_param_args_(op, q_param)
    return q_param


def create_int8_given_tensor_fill(tensor, out_blob_name, preserve_sparsity=False):
    """
    Create Int8GivenTensorFill op that quantizes the given tensor and outputs
    an Int8Tensor with out_blob_name.
    """
    op = core.CreateOperator("Int8GivenTensorFill", [], out_blob_name)
    q_param = add_quantization_param_args(op, tensor, preserve_sparsity)
    quantized_tensor = (
        np.around(tensor / q_param.scale).astype(np.int32) + q_param.zero_point
    )
    quantized_tensor = np.maximum(0, np.minimum(quantized_tensor, 255))
    op.arg.extend(
        [
            utils.MakeArgument("values", quantized_tensor.astype(np.uint8).tobytes()),
            utils.MakeArgument("shape", quantized_tensor.shape),
        ]
    )
    return op, q_param


def create_int8_bias_tensor_fill(tensor, out_blob_name, x_q_param, w_q_param):
    """
    Similar to create_int8_given_tensor_fill, but for bias blobs to be stored
    as int32.
    """
    scale = x_q_param.scale * w_q_param.scale
    quantized_tensor = np.around(tensor / scale).astype(np.int32)
    quantized_tensor.reshape(-1)
    op = core.CreateOperator("Int8GivenIntTensorFill", [], out_blob_name)
    op.arg.extend(
        [
            utils.MakeArgument("values", quantized_tensor),
            utils.MakeArgument("shape", quantized_tensor.shape),
        ]
    )
    q_param = hardcode_scale_zp.QuantizationParam(scale, 0)
    add_quantization_param_args_(op, q_param)
    return op
