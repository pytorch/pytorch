## @package algebra
# Module caffe2.python.helpers.algebra






def transpose(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """Transpose."""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.Transpose(blob_in, blob_out, **kwargs)


def sum(model, blob_in, blob_out, **kwargs):
    """Sum"""
    return model.net.Sum(blob_in, blob_out, **kwargs)


def reduce_sum(model, blob_in, blob_out, **kwargs):
    """ReduceSum"""
    return model.net.ReduceSum(blob_in, blob_out, **kwargs)


def sub(model, blob_in, blob_out, **kwargs):
    """Subtract"""
    return model.net.Sub(blob_in, blob_out, **kwargs)


def mat_mul(model, blob_in, blob_out, **kwargs):
    """Matrix multiplication"""
    return model.net.MatMul(blob_in, blob_out, **kwargs)


def arg_min(model, blob_in, blob_out, **kwargs):
    """ArgMin"""
    return model.net.ArgMin(blob_in, blob_out, **kwargs)

def batch_mat_mul(model, blob_in, blob_out,
                  enable_tensor_core=False, **kwargs):
    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    return model.net.BatchMatMul(blob_in, blob_out, **kwargs)

def sparse_lengths_sum_4bit_rowwise_sparse(model, blob_in, blob_out, **kwargs):
    return model.net.SparseLengthsSum4BitRowwiseSparse(blob_in, blob_out, **kwargs)
