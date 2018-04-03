## @package fc
# Module caffe2.python.helpers.fc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags


def _FC_or_packed_FC(
    model, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None,
        bias_init=None, WeightInitializer=None, BiasInitializer=None,
        enable_tensor_core=False, float16_compute=False, **kwargs
):
    WeightInitializer = initializers.update_initializer(
        WeightInitializer, weight_init, ("XavierFill", {})
    )
    BiasInitializer = initializers.update_initializer(
        BiasInitializer, bias_init, ("ConstantFill", {})
    )
    if not model.init_params:
        WeightInitializer = initializers.ExternalInitializer()
        BiasInitializer = initializers.ExternalInitializer()

    blob_out = blob_out or model.net.NextName()
    bias_tags = [ParameterTags.BIAS]
    if 'freeze_bias' in kwargs:
        bias_tags.append(ParameterTags.COMPUTED_PARAM)

    weight = model.create_param(
        param_name=blob_out + '_w',
        shape=[dim_out, dim_in],
        initializer=WeightInitializer,
        tags=ParameterTags.WEIGHT
    )
    bias = model.create_param(
        param_name=blob_out + '_b',
        shape=[dim_out, ],
        initializer=BiasInitializer,
        tags=bias_tags
    )

    # enable TensorCore by setting appropriate engine
    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    # Enable float 16 compute kernel (relevant for CUDA)
    if float16_compute:
        kwargs['float16_compute'] = True

    return op_call([blob_in, weight, bias], blob_out, **kwargs)


def fc(model, *args, **kwargs):
    return _FC_or_packed_FC(model, model.net.FC, *args, **kwargs)


def packed_fc(model, *args, **kwargs):
    return _FC_or_packed_FC(model, model.net.PackedFC, *args, **kwargs)


def fc_decomp(
    model, blob_in, blob_out, dim_in, dim_out,
    rank_approx=5, weight_init=None, bias_init=None,
    WeightInitializer=None, BiasInitializer=None, **kwargs
):
    """FC_Decomp version
    Here we assume that the rank of original input is bigger than 5.
    """
    WeightInitializer = initializers.update_initializer(
        WeightInitializer, weight_init, ("XavierFill", {})
    )
    BiasInitializer = initializers.update_initializer(
        BiasInitializer, bias_init, ("ConstantFill", {})
    )
    blob_out = blob_out or model.net.NextName()
    u = model.create_param(
        param_name=blob_out + '_u',
        shape=[dim_out, rank_approx],
        initializer=WeightInitializer,
    )
    v = model.create_param(
        param_name=blob_out + '_v',
        shape=[dim_in, rank_approx],
        initializer=WeightInitializer,
    )
    bias = model.create_param(
        param_name=blob_out + '_b',
        shape=[dim_out, ],
        initializer=BiasInitializer,
    )
    return model.net.FC_Decomp([blob_in, u, v, bias], blob_out, **kwargs)


def fc_prune(
    model, blob_in, blob_out, dim_in, dim_out,
    weight_init=None, bias_init=None, mask_init=None,
    threshold=0.00001, need_compress_rate=False,
    comp_lb=0.05,
    **kwargs
):
    """FC_Prune version
    Runnable so far. Great!:)
    """
    weight_init = weight_init if weight_init else ('XavierFill', {})
    bias_init = bias_init if bias_init else ('ConstantFill', {})
    mask_init = mask_init if mask_init else ('ConstantFill', {})
    blob_out = blob_out or model.net.NextName()
    compress_rate = blob_out + '_compress_rate'
    if model.init_params:
        compress_lb = model.param_init_net.ConstantFill(
            [],
            blob_out + '_lb',
            shape=[1],
            value=comp_lb
        )
        weight = model.param_init_net.__getattr__(weight_init[0])(
            [],
            blob_out + '_w',
            shape=[dim_out, dim_in],
            **weight_init[1]
        )
        mask = model.param_init_net.ConstantFill(
            [],
            blob_out + '_m',
            shape=[dim_out, dim_in],
            value=1.0
        )
        ag_dw = model.param_init_net.__getattr__(mask_init[0])(
            [],
            blob_out + '_ag_dw',
            shape=[dim_out, dim_in],
            **mask_init[1]
        )
        bias = model.param_init_net.__getattr__(bias_init[0])(
            [],
            blob_out + '_b',
            shape=[dim_out, ],
            **bias_init[1]
        )
        mask_seq = model.param_init_net.__getattr__(mask_init[0])(
            [],
            blob_out + '_mask_seq',
            shape=[dim_out, dim_in],
            **mask_init[1]
        )
        thres = model.param_init_net.ConstantFill(
            [],
            blob_out + '_thres',
            shape=[1],
            value=threshold
        )
    else:
        compress_lb = core.ScopedBlobReference(
            blob_out + '_lb', model.param_init_net)
        weight = core.ScopedBlobReference(
            blob_out + '_w', model.param_init_net)
        bias = core.ScopedBlobReference(
            blob_out + '_b', model.param_init_net)
        mask = core.ScopedBlobReference(
            blob_out + '_m', model.param_init_net)
        ag_dw = core.ScopedBlobReference(
            blob_out + '_ag_dw', model.param_init_net)
        mask_seq = core.ScopedBlobReference(
            blob_out + '_mask_seq', model.param_init_net)
        thres = core.ScopedBlobReference(
            blob_out + '_thres', model.param_init_net)

    model.AddParameter(weight)
    model.AddParameter(bias)
    if need_compress_rate:
        return model.net.FC_Prune([blob_in, weight, mask, bias, ag_dw, mask_seq,
                                   thres, compress_lb],
                                  [blob_out, compress_rate], **kwargs)
    else:
        return model.net.FC_Prune([blob_in, weight, mask,
                                   bias, ag_dw, mask_seq,
                                   thres, compress_lb],
                                  blob_out, **kwargs)


def fc_sparse(
    model, blob_in, blob_out, w_csr, iw, jw, bias,
    **kwargs
):
    """FC_Sparse: Only takes in alocated weights"""
    if not (w_csr and iw and jw and bias):
        print("Warning...")
    model.AddParameter(w_csr)
    model.AddParameter(iw)
    model.AddParameter(jw)
    model.AddParameter(bias)
    return model.net.FC_Sparse([blob_in, w_csr, iw, jw, bias],
                               blob_out, **kwargs)
