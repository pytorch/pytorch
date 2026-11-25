"""CuTeDSL implementation of ``aten::_scaled_mm_mxfp6``.

The CUDA ``_scaled_mm_v2`` kernel validates MXFP6 (``torch.float6_e2m3fn_x4``)
calls and hands them to ``_scaled_mm_mxfp6.out``, whose aten kernel only reports
that this override is missing. Other dtypes never reach Python.
"""

from ... import cutedsl_utils as cu


def _impl(*args, **kwargs):
    # Imported on the first MXFP6 call, so registration never loads CuTeDSL.
    from .mxfp6_kernel import mxfp6_scaled_mm

    return mxfp6_scaled_mm(*args, **kwargs)


def register_to_dispatch() -> None:
    for op_symbol in ("_scaled_mm_mxfp6", "_scaled_mm_mxfp6.out"):
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=None,
            impl=_impl,
            unconditional_override=True,
        )
