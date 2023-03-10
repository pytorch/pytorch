"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

import functools
from typing import Optional, Sequence

import torch
from torch import _C
from torch.onnx import _type_utils, errors, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = ["layer_norm", "stft"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
):
    # normalized_shape: input shape from an expected input of size
    # axis: The first normalization dimension.
    # layer_norm normalizes on the last D dimensions,
    # where D is the size of normalized_shape
    axis = -len(normalized_shape)
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )


def _compute_edge_sizes(n_fft, window_size):
    """Helper function to compute the sizes of the edges (left and right)
    of a given window centered within an FFT size."""
    left = (n_fft - window_size) // 2
    right = n_fft - left - window_size
    return left, right


@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b")
@_beartype.beartype
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[_C.Value] = None,
    normalized: bool = False,
    onesided: Optional[bool] = True,
    return_complex: Optional[bool] = False,
) -> _C.Value:
    """Associates `torch.stft` with the `STFT` ONNX operator.
    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Args:
        g: Graph to write the ONNX representation into
        input: Input tensor for the transformation
        n_fft: FFT size
        hop_length: Size of the hop. Defaults to `floot(n_fft // 4)`
        win_length: Size of the analysis window. Defaults to `n_fft`
        window: Analysis window. Defaults to a window of all ones
        normalized: Whether to return a normalized STFT
        onesided: Whether to return only half (+1) of the results, given the
            symmetry of the STFT
        return_complex: Whether to return the complex value (Note: Must be
            `False` or `None`)

    Returns:
        op: Operator for torch.stft associated with STFT (ONNX)
    """
    # Checks
    if return_complex:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support complex types", value=input
        )

    # Get STFT sizes
    frame_step_value = hop_length if hop_length is not None else n_fft // 4
    frame_step_const = g.op(
        "Constant", value_t=torch.tensor(frame_step_value, dtype=torch.int64)
    )
    frame_length_const = g.op(
        "Constant", value_t=torch.tensor(n_fft, dtype=torch.int64)
    )

    # Pre-process input if needed
    signal = input
    signal_rank = symbolic_helper._get_tensor_rank(signal)
    if signal_rank == 1:
        # Add batch dimension
        signal = g.op(
            "Unsqueeze",
            signal,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
    elif signal_rank > 2:
        raise errors.SymbolicValueError(
            msg="STFT can only take inputs of 1 [signal] or 2 [batch, signal] dimensions. "
            f"Current rank of signal is {signal_rank}, please reduce it.",
            value=input,
        )

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    n_win = symbolic_helper._get_tensor_dim_size(window, dim=0)
    if n_win is not None:
        win_length_default = win_length if win_length else n_fft
        assert n_win == win_length_default, (
            "Analysis window size must equal `win_length` or `n_fft`. "
            f"Please, set `win_length` or `n_fft` to match `window` size ({n_win})",
        )

        # Center window around zeros if needed (required by ONNX's STFT)
        if n_win < n_fft:
            left, right = _compute_edge_sizes(n_fft, n_win)
            left_win = g.op("Constant", value_t=torch.zeros((left)))
            right_win = g.op("Constant", value_t=torch.zeros((right)))
            window = g.op("Concat", left_win, window, right_win, axis_i=0)

    # Create window, if needed
    if symbolic_helper._is_none(window):
        if win_length:
            if win_length > n_fft:
                raise errors.SymbolicValueError(
                    msg="The analysis window can't be longer than the size of the FFT. "
                    f"Please set `win_length` ({win_length}) to `n_fft` ({n_fft}) or less.",
                    value=input,
                )

            # Center window, if needed
            left, right = _compute_edge_sizes(n_fft, win_length)
            torch_window = torch.hstack(
                (torch.zeros((left)), torch.ones((win_length)), torch.zeros((right)))
            )
        else:
            # Rectangle window
            torch_window = torch.ones((n_fft))
        assert torch_window.shape[0] == n_fft
        window = g.op("Constant", value_t=torch_window)
    window = g.op(
        "Cast", window, to_i=_type_utils.JitScalarType.from_value(signal).onnx_type()
    )

    # Run STFT
    result = g.op(
        "STFT",
        signal,
        frame_step_const,
        window,
        frame_length_const,
        onesided_i=1 if onesided is None or onesided else 0,
    )

    # Transpose to mimic torch.stft's behavior
    result = g.op("Transpose", result, perm_i=[0, 2, 1, 3])

    # Remove batch dimension, if needed
    if signal_rank == 1:
        result = g.op(
            "Squeeze",
            result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # Normalize, if needed
    if normalized:
        sqrt_nfft = torch.sqrt(torch.tensor(n_fft, dtype=signal.type().dtype()))
        result = g.op("Div", result, g.op("Constant", value_t=sqrt_nfft))

    return result
