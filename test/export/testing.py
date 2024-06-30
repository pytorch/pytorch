import functools
import unittest
from unittest.mock import patch

import torch

aten = torch.ops.aten

# This list is not meant to be comprehensive
_COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY = [
    aten.arctan2.default,
    aten.divide.Tensor,
    aten.divide.Scalar,
    aten.divide.Tensor_mode,
    aten.divide.Scalar_mode,
    aten.multiply.Tensor,
    aten.multiply.Scalar,
    aten.subtract.Tensor,
    aten.subtract.Scalar,
    aten.true_divide.Tensor,
    aten.true_divide.Scalar,
    aten.greater.Tensor,
    aten.greater.Scalar,
    aten.greater_equal.Tensor,
    aten.greater_equal.Scalar,
    aten.less_equal.Tensor,
    aten.less_equal.Scalar,
    aten.less.Tensor,
    aten.less.Scalar,
    aten.not_equal.Tensor,
    aten.not_equal.Scalar,
    aten.cat.names,
    aten.sum.dim_DimnameList,
    aten.mean.names_dim,
    aten.prod.dim_Dimname,
    aten.all.dimname,
    aten.norm.names_ScalarOpt_dim,
    aten.norm.names_ScalarOpt_dim_dtype,
    aten.var.default,
    aten.var.dim,
    aten.var.names_dim,
    aten.var.correction_names,
    aten.std.default,
    aten.std.dim,
    aten.std.names_dim,
    aten.std.correction_names,
    aten.absolute.default,
    aten.arccos.default,
    aten.arccosh.default,
    aten.arcsin.default,
    aten.arcsinh.default,
    aten.arctan.default,
    aten.arctanh.default,
    aten.clip.default,
    aten.clip.Tensor,
    aten.fix.default,
    aten.negative.default,
    aten.square.default,
    aten.size.int,
    aten.size.Dimname,
    aten.stride.int,
    aten.stride.Dimname,
    aten.repeat_interleave.self_Tensor,
    aten.repeat_interleave.self_int,
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.atleast_1d.Sequence,
    aten.atleast_2d.Sequence,
    aten.atleast_3d.Sequence,
    aten.linear.default,
    aten.conv2d.default,
    aten.conv2d.padding,
    aten.mish_backward.default,
    aten.silu_backward.default,
    aten.index_add.dimname,
    aten.pad_sequence.default,
    aten.index_copy.dimname,
    aten.upsample_nearest1d.vec,
    aten.upsample_nearest2d.vec,
    aten.upsample_nearest3d.vec,
    aten._upsample_nearest_exact1d.vec,
    aten._upsample_nearest_exact2d.vec,
    aten._upsample_nearest_exact3d.vec,
    aten.rnn_tanh.input,
    aten.rnn_tanh.data,
    aten.rnn_relu.input,
    aten.rnn_relu.data,
    aten.lstm.input,
    aten.lstm.data,
    aten.gru.input,
    aten.gru.data,
    aten._upsample_bilinear2d_aa.vec,
    aten._upsample_bicubic2d_aa.vec,
    aten.upsample_bilinear2d.vec,
    aten.upsample_trilinear3d.vec,
    aten.upsample_linear1d.vec,
    aten.matmul.default,
    aten.upsample_bicubic2d.vec,
    aten.__and__.Scalar,
    aten.__and__.Tensor,
    aten.__or__.Tensor,
    aten.__or__.Scalar,
    aten.__xor__.Tensor,
    aten.__xor__.Scalar,
    aten.scatter.dimname_src,
    aten.scatter.dimname_value,
    aten.scatter_add.dimname,
    aten.is_complex.default,
    aten.logsumexp.names,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
    aten.where.Scalar,
    aten.where.default,
    aten.item.default,
    aten.any.dimname,
    aten.std_mean.default,
    aten.std_mean.dim,
    aten.std_mean.names_dim,
    aten.std_mean.correction_names,
    aten.var_mean.default,
    aten.var_mean.dim,
    aten.var_mean.names_dim,
    aten.var_mean.correction_names,
    aten.broadcast_tensors.default,
    aten.stft.default,
    aten.stft.center,
    aten.istft.default,
    aten.index_fill.Dimname_Scalar,
    aten.index_fill.Dimname_Tensor,
    aten.index_select.dimname,
    aten.diag.default,
    aten.cumsum.dimname,
    aten.cumprod.dimname,
    aten.meshgrid.default,
    aten.meshgrid.indexing,
    aten.fft_fft.default,
    aten.fft_ifft.default,
    aten.fft_rfft.default,
    aten.fft_irfft.default,
    aten.fft_hfft.default,
    aten.fft_ihfft.default,
    aten.fft_fftn.default,
    aten.fft_ifftn.default,
    aten.fft_rfftn.default,
    aten.fft_ihfftn.default,
    aten.fft_irfftn.default,
    aten.fft_hfftn.default,
    aten.fft_fft2.default,
    aten.fft_ifft2.default,
    aten.fft_rfft2.default,
    aten.fft_irfft2.default,
    aten.fft_hfft2.default,
    aten.fft_ihfft2.default,
    aten.fft_fftshift.default,
    aten.fft_ifftshift.default,
    aten.selu.default,
    aten.margin_ranking_loss.default,
    aten.hinge_embedding_loss.default,
    aten.nll_loss.default,
    aten.prelu.default,
    aten.relu6.default,
    aten.pairwise_distance.default,
    aten.pdist.default,
    aten.special_ndtr.default,
    aten.cummax.dimname,
    aten.cummin.dimname,
    aten.logcumsumexp.dimname,
    aten.max.other,
    aten.max.names_dim,
    aten.min.other,
    aten.min.names_dim,
    aten.linalg_eigvals.default,
    aten.median.names_dim,
    aten.nanmedian.names_dim,
    aten.mode.dimname,
    aten.gather.dimname,
    aten.sort.dimname,
    aten.sort.dimname_stable,
    aten.argsort.default,
    aten.argsort.dimname,
    aten.rrelu.default,
    aten.conv_transpose1d.default,
    aten.conv_transpose2d.input,
    aten.conv_transpose3d.input,
    aten.conv1d.default,
    aten.conv1d.padding,
    aten.conv3d.default,
    aten.conv3d.padding,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Scalar,
    aten.ldexp.Tensor,
    aten._version.default,
]


def make_test_cls_with_mocked_export(
    cls, cls_prefix, fn_suffix, mocked_export_fn, xfail_prop=None
):
    MockedTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    MockedTestClass.__qualname__ = MockedTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(MockedTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_mocked_export(fn, mocked_export_fn)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(MockedTestClass, new_name, new_fn)
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(MockedTestClass, name):
            setattr(MockedTestClass, name, getattr(cls, name))

    return MockedTestClass


def _make_fn_with_mocked_export(fn, mocked_export_fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        try:
            from . import test_export
        except ImportError:
            import test_export

        with patch(f"{test_export.__name__}.export", mocked_export_fn):
            return fn(*args, **kwargs)

    return _fn


# Controls tests generated in test/export/test_export_training_ir_to_run_decomp.py
def expectedFailureTrainingIRToRunDecomp(fn):
    fn._expected_failure_training_ir_to_run_decomp = True
    return fn


# Controls tests generated in test/export/test_export_nonstrict.py
def expectedFailureNonStrict(fn):
    fn._expected_failure_non_strict = True
    return fn


# Controls tests generated in test/export/test_retraceability.py
def expectedFailureRetraceability(fn):
    fn._expected_failure_retrace = True
    return fn


# Controls tests generated in test/export/test_serdes.py
def expectedFailureSerDer(fn):
    fn._expected_failure_serdes = True
    return fn


def expectedFailureSerDerPreDispatch(fn):
    fn._expected_failure_serdes_pre_dispatch = True
    return fn


def expectedFailurePreDispatchRunDecomp(fn):
    fn._expected_failure_pre_dispatch = True
    return fn
