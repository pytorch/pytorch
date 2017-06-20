## @package conv
# Module caffe2.python.helpers.conv
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

def _ConvBase(
    model,
    is_nd,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    WeightInitializer=None,
    BiasInitializer=None,
    group=1,
    transform_inputs=None,
    use_cudnn=False,
    order="NCHW",
    cudnn_exhaustive_search=False,
    ws_nbytes_limit=None,
    **kwargs
):
    kernels = []
    if is_nd:
        if not isinstance(kernel, list):
            kernels = [kernel]
        else:
            kernels = kernel
    else:
        kernels = [kernel] * 2

    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
        kwargs['exhaustive_search'] = cudnn_exhaustive_search
        if ws_nbytes_limit:
            kwargs['ws_nbytes_limit'] = ws_nbytes_limit

    use_bias =\
            False if ("no_bias" in kwargs and kwargs["no_bias"]) else True
    blob_out = blob_out or model.net.NextName()
    weight_shape = [dim_out]
    if order == "NCHW":
        weight_shape.append(int(dim_in / group))
        weight_shape.extend(kernels)
    else:
        weight_shape.extend(kernels)
        weight_shape.append(int(dim_in / group))

    WeightInitializer = initializers.update_initializer(
        WeightInitializer, weight_init, ("XavierFill", {})
    )
    BiasInitializer = initializers.update_initializer(
        BiasInitializer, bias_init, ("ConstantFill", {})
    )
    if not model.init_params:
        WeightInitializer = initializers.ExternalInitializer()
        BiasInitializer = initializers.ExternalInitializer()

    weight = model.create_param(
        param_name=blob_out + '_w',
        shape=weight_shape,
        initializer=WeightInitializer,
        tags=ParameterTags.WEIGHT
    )
    if use_bias:
        bias = model.create_param(
            param_name=blob_out + '_b',
            shape=[dim_out, ],
            initializer=BiasInitializer,
            tags=ParameterTags.BIAS
        )

    if use_bias:
        inputs = [blob_in, weight, bias]
    else:
        inputs = [blob_in, weight]

    if transform_inputs is not None:
        transform_inputs(model, blob_out, inputs)

    # For the operator, we no longer need to provide the no_bias field
    # because it can automatically figure this out from the number of
    # inputs.
    if 'no_bias' in kwargs:
        del kwargs['no_bias']
    if group != 1:
        kwargs['group'] = group
    if is_nd:
        return model.net.Conv(
            inputs,
            blob_out,
            kernels=kernels,
            order=order,
            **kwargs)
    else:
        return model.net.Conv(
            inputs,
            blob_out,
            kernel=kernel,
            order=order,
            **kwargs)


def conv_nd(
    model,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    WeightInitializer=None,
    BiasInitializer=None,
    group=1,
    transform_inputs=None,
    order="NCHW",
    **kwargs
):
    """N-dimensional convolution for inputs with NCHW storage order.
    """
    assert order == "NCHW", "ConvNd only supported for NCHW storage."
    return _ConvBase(model, True, blob_in, blob_out, dim_in, dim_out, kernel,
                     weight_init, bias_init, WeightInitializer, BiasInitializer,
                     group, transform_inputs, order=order, **kwargs)


def conv(
    model,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    WeightInitializer=None,
    BiasInitializer=None,
    group=1,
    transform_inputs=None,
    **kwargs
):
    """2-dimensional convolution.
    """
    return _ConvBase(model, False, blob_in, blob_out, dim_in, dim_out, kernel,
                     weight_init, bias_init, WeightInitializer, BiasInitializer,
                     group, transform_inputs, **kwargs)


def conv_transpose(
    model,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    use_cudnn=False,
    order="NCHW",
    cudnn_exhaustive_search=False,
    ws_nbytes_limit=None,
    **kwargs
):
    """ConvTranspose.
    """
    weight_init = weight_init if weight_init else ('XavierFill', {})
    bias_init = bias_init if bias_init else ('ConstantFill', {})
    blob_out = blob_out or model.net.NextName()
    weight_shape = (
        [dim_in, dim_out, kernel, kernel]
        if order == "NCHW" else [dim_in, kernel, kernel, dim_out]
    )
    if model.init_params:
        weight = model.param_init_net.__getattr__(weight_init[0])(
            [],
            blob_out + '_w',
            shape=weight_shape,
            **weight_init[1]
        )
        bias = model.param_init_net.__getattr__(bias_init[0])(
            [],
            blob_out + '_b',
            shape=[dim_out, ],
            **bias_init[1]
        )
    else:
        weight = core.ScopedBlobReference(
            blob_out + '_w', model.param_init_net)
        bias = core.ScopedBlobReference(
            blob_out + '_b', model.param_init_net)
    model.AddParameter(weight, ParameterTags.WEIGHT)
    model.AddParameter(bias, ParameterTags.BIAS)
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
        kwargs['exhaustive_search'] = cudnn_exhaustive_search
        if ws_nbytes_limit:
            kwargs['ws_nbytes_limit'] = ws_nbytes_limit
    return model.net.ConvTranspose(
        [blob_in, weight, bias],
        blob_out,
        kernel=kernel,
        order=order,
        **kwargs
    )


def group_conv(
    model,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    group=1,
    **kwargs
):
    """Group Convolution.

    This is essentially the same as Conv with a group argument passed in.
    We specialize this for backward interface compatibility.
    """
    return conv(model, blob_in, blob_out, dim_in, dim_out, kernel,
                weight_init=weight_init, bias_init=bias_init,
                group=group, **kwargs)


def group_conv_deprecated(
    model,
    blob_in,
    blob_out,
    dim_in,
    dim_out,
    kernel,
    weight_init=None,
    bias_init=None,
    group=1,
    use_cudnn=False,
    order="NCHW",
    cudnn_exhaustive_search=False,
    ws_nbytes_limit=None,
    **kwargs
):
    """GroupConvolution's deprecated interface.

    This is used to simulate a group convolution via split and concat. You
    should always use the new group convolution in your new code.
    """
    weight_init = weight_init if weight_init else ('XavierFill', {})
    bias_init = bias_init if bias_init else ('ConstantFill', {})
    use_bias = False if ("no_bias" in kwargs and kwargs["no_bias"]) else True
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
        kwargs['exhaustive_search'] = cudnn_exhaustive_search
        if ws_nbytes_limit:
            kwargs['ws_nbytes_limit'] = ws_nbytes_limit
            if dim_in % group:
                raise ValueError("dim_in should be divisible by group.")
    if dim_out % group:
        raise ValueError("dim_out should be divisible by group.")
    splitted_blobs = model.net.DepthSplit(
        blob_in,
        ['_' + blob_out + '_gconv_split_' + str(i) for i in range(group)],
        dimensions=[int(dim_in / group) for i in range(group)],
        order=order
    )
    weight_shape = (
        [dim_out / group, dim_in / group, kernel, kernel]
        if order == "NCHW" else
        [dim_out / group, kernel, kernel, dim_in / group]
    )
    # Make sure that the shapes are of int format. Especially for py3 where
    # int division gives float output.
    weight_shape = [int(v) for v in weight_shape]
    conv_blobs = []
    for i in range(group):
        if model.init_params:
            weight = model.param_init_net.__getattr__(weight_init[0])(
                [],
                blob_out + '_gconv_%d_w' % i,
                shape=weight_shape,
                **weight_init[1]
            )
            if use_bias:
                bias = model.param_init_net.__getattr__(bias_init[0])(
                    [],
                    blob_out + '_gconv_%d_b' % i,
                    shape=[int(dim_out / group)],
                    **bias_init[1]
                )
        else:
            weight = core.ScopedBlobReference(
                blob_out + '_gconv_%d_w' % i, model.param_init_net)
            if use_bias:
                bias = core.ScopedBlobReference(
                    blob_out + '_gconv_%d_b' % i, model.param_init_net)
        model.AddParameter(weight, ParameterTags.WEIGHT)
        if use_bias:
            model.AddParameter(bias, ParameterTags.BIAS)
        if use_bias:
            inputs = [weight, bias]
        else:
            inputs = [weight]
        if 'no_bias' in kwargs:
            del kwargs['no_bias']
        conv_blobs.append(
            splitted_blobs[i].Conv(
                inputs,
                blob_out + '_gconv_%d' % i,
                kernel=kernel,
                order=order,
                **kwargs
            )
        )
    concat, concat_dims = model.net.Concat(
        conv_blobs,
        [blob_out,
         "_" + blob_out + "_concat_dims"],
        order=order
    )
    return concat
