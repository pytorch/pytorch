# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR recurrent translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import residual


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten


def _rnn_output_to_layer_output(
    op: TranslationContext, Y: ir.Value, bidirectional: bool, hidden_size: int
) -> ir.Value:
    """Convert [seq, directions, batch, hidden] to [seq, batch, directions * hidden]."""
    builder = op.builder
    if not bidirectional:
        return op.Squeeze(Y, [1])
    Y = op.Transpose(Y, perm=[0, 2, 1, 3])
    # Zeros copy the sequence and batch dimensions without runtime shape computation.
    return op.Reshape(
        Y, builder.constant([0, 0, 2 * hidden_size], dtype=ir.DataType.INT64)
    )


@onnx_impl(aten.gru.input)
def aten_gru(
    op: TranslationContext,
    input: ir.Value,
    hx: ir.Value,
    params: Sequence[ir.Value],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[ir.Value, ir.Value]:
    """gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"""
    if hx.shape is None or len(hx.shape) != 3 or not isinstance(hx.shape[2], int):
        raise ValueError(
            "GRU requires a known hidden size in hx.shape[2]. Keep the hidden dimension static when exporting."
        )
    if num_layers < 1:
        raise ValueError(
            f"GRU requires at least one layer, got {num_layers}. Set num_layers >= 1."
        )

    # Determine number of directions
    num_directions = 2 if bidirectional else 1

    # Get dimensions
    if batch_first:
        # Convert from [batch, seq, input_size] to [seq, batch, input_size]
        input = op.Transpose(input, perm=[1, 0, 2])

    hidden_size = op.Shape(hx, start=2, end=3)

    # Process each layer
    current_input = input
    output_h_list = []

    for layer_idx in range(num_layers):
        # Extract hidden state for this layer
        layer_start = layer_idx * num_directions
        layer_end = (layer_idx + 1) * num_directions
        layer_h = op.Slice(hx, [layer_start], [layer_end], [0])

        # Extract parameters for this layer
        # Parameter layout: [W_ih, W_hh, b_ih, b_hh] for each direction
        params_per_direction = 4 if has_biases else 2
        params_per_layer = params_per_direction * num_directions
        param_start_idx = layer_idx * params_per_layer

        # Build weight matrices for ONNX GRU
        # ONNX expects: W[zrh] shape [num_directions, 3*hidden_size, input_size]
        # PyTorch provides: W_ih shape [3*hidden_size, input_size]
        W_list = []
        R_list = []
        B_list = [] if has_biases else None

        for dir_idx in range(num_directions):
            dir_param_start = param_start_idx + dir_idx * params_per_direction
            W_ih = params[
                dir_param_start
            ]  # [3*hidden_size, input_size] - PyTorch order: [r,z,n]
            W_hh = params[
                dir_param_start + 1
            ]  # [3*hidden_size, hidden_size] - PyTorch order: [r,z,n]

            # Reorder gates from PyTorch [r,z,n] to ONNX [z,r,n]
            # Split into individual gates
            W_ir = op.Slice(W_ih, [0], hidden_size, [0])
            W_iz = op.Slice(W_ih, hidden_size, op.Mul(hidden_size, 2), [0])
            W_in = op.Slice(
                W_ih,
                op.Mul(hidden_size, 2),
                op.Mul(hidden_size, 3),
                [0],
            )

            W_hr = op.Slice(W_hh, [0], hidden_size, [0])
            W_hz = op.Slice(W_hh, hidden_size, op.Mul(hidden_size, 2), [0])
            W_hn = op.Slice(
                W_hh,
                op.Mul(hidden_size, 2),
                op.Mul(hidden_size, 3),
                [0],
            )

            # Reorder: [z,r,n]
            W_ih_reordered = op.Concat(
                W_iz, W_ir, W_in, axis=0
            )  # [3*hidden_size, input_size] - ONNX order
            W_hh_reordered = op.Concat(
                W_hz, W_hr, W_hn, axis=0
            )  # [3*hidden_size, hidden_size] - ONNX order

            # Add direction dimension
            W_ih_expanded = op.Unsqueeze(
                W_ih_reordered, [0]
            )  # [1, 3*hidden_size, input_size]
            W_hh_expanded = op.Unsqueeze(
                W_hh_reordered, [0]
            )  # [1, 3*hidden_size, hidden_size]

            W_list.append(W_ih_expanded)
            R_list.append(W_hh_expanded)

            if has_biases:
                b_ih = params[
                    dir_param_start + 2
                ]  # [3*hidden_size] - PyTorch order: [r,z,n]
                b_hh = params[
                    dir_param_start + 3
                ]  # [3*hidden_size] - PyTorch order: [r,z,n]

                # Reorder biases from PyTorch [r,z,n] to ONNX [z,r,n]
                b_ir = op.Slice(b_ih, [0], hidden_size, [0])
                b_iz = op.Slice(
                    b_ih,
                    hidden_size,
                    op.Mul(hidden_size, 2),
                    [0],
                )
                b_in = op.Slice(
                    b_ih,
                    op.Mul(hidden_size, 2),
                    op.Mul(hidden_size, 3),
                    [0],
                )

                b_hr = op.Slice(b_hh, [0], hidden_size, [0])
                b_hz = op.Slice(
                    b_hh,
                    hidden_size,
                    op.Mul(hidden_size, 2),
                    [0],
                )
                b_hn = op.Slice(
                    b_hh,
                    op.Mul(hidden_size, 2),
                    op.Mul(hidden_size, 3),
                    [0],
                )

                # Reorder: [z,r,n]
                b_ih_reordered = op.Concat(
                    b_iz, b_ir, b_in, axis=0
                )  # [3*hidden_size] - ONNX order
                b_hh_reordered = op.Concat(
                    b_hz, b_hr, b_hn, axis=0
                )  # [3*hidden_size] - ONNX order

                # ONNX expects biases concatenated: [Wb[zrh], Rb[zrh]]
                b_combined = op.Concat(
                    b_ih_reordered, b_hh_reordered, axis=0
                )  # [6*hidden_size]
                b_expanded = op.Unsqueeze(b_combined, [0])  # [1, 6*hidden_size]
                B_list.append(b_expanded)

        # Concatenate weights for all directions
        W = op.Concat(*W_list, axis=0) if len(W_list) > 1 else W_list[0]
        R = op.Concat(*R_list, axis=0) if len(R_list) > 1 else R_list[0]
        B = (
            op.Concat(*B_list, axis=0)
            if has_biases and len(B_list) > 1
            else (B_list[0] if has_biases else None)
        )

        # Call ONNX GRU operator
        direction = "bidirectional" if bidirectional else "forward"

        # Extract hidden_size from hx shape: [num_layers * num_directions, batch, hidden_size]
        hidden_size_attr = hx.shape[2]

        # linear_before_reset=1 matches PyTorch's GRU formulation where the linear
        # transformation is applied before multiplying by the reset gate:
        #   ht = g(Xt*(Wh^T) + rt (.) (Ht-1*(Rh^T) + Rbh) + Wbh)
        # The ONNX default (linear_before_reset=0) uses a different equation and
        # would produce numerically incorrect results.
        if B is not None:
            Y, Y_h = op.GRU(
                current_input,
                W,
                R,
                B,
                None,
                layer_h,
                direction=direction,
                hidden_size=hidden_size_attr,
                linear_before_reset=1,
                outputs=2,
            )
        else:
            Y, Y_h = op.GRU(
                current_input,
                W,
                R,
                None,
                None,
                layer_h,
                direction=direction,
                hidden_size=hidden_size_attr,
                linear_before_reset=1,
                outputs=2,
            )

        current_input = _rnn_output_to_layer_output(
            op, Y, bidirectional, hidden_size_attr
        )

        # Apply dropout if not last layer and dropout > 0
        if layer_idx < num_layers - 1 and dropout > 0.0 and train:
            current_input = residual.aten_dropout(op, current_input, dropout, train)

        # Store final hidden state
        output_h_list.append(Y_h)

    # Concatenate all layer outputs
    final_h = (
        output_h_list[0]
        if len(output_h_list) == 1
        else op.Concat(*output_h_list, axis=0)
    )

    # Handle batch_first for output
    if batch_first:
        # Convert from [seq, batch, features] to [batch, seq, features]
        current_input = op.Transpose(current_input, perm=[1, 0, 2])

    return current_input, final_h


@onnx_impl(aten.lstm.input)
def aten_lstm(
    op: TranslationContext,
    input: ir.Value,
    hx: Sequence[ir.Value],
    params: Sequence[ir.Value],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)"""

    # Extract initial hidden and cell states
    initial_h = hx[0]  # Shape: [num_directions * num_layers, batch_size, hidden_size]
    initial_c = hx[1]  # Shape: [num_directions * num_layers, batch_size, hidden_size]
    if (
        initial_h.shape is None
        or len(initial_h.shape) != 3
        or not isinstance(initial_h.shape[2], int)
    ):
        raise ValueError(
            "LSTM requires a known hidden size in hx[0].shape[2]. Keep the hidden dimension static when exporting."
        )
    if initial_c.shape is not None and initial_h.shape[2] != initial_c.shape[2]:
        raise NotImplementedError(
            "LSTM projections are not supported by ONNX LSTM. Export an LSTM with proj_size=0."
        )
    if num_layers < 1:
        raise ValueError(
            f"LSTM requires at least one layer, got {num_layers}. Set num_layers >= 1."
        )

    # Determine number of directions
    num_directions = 2 if bidirectional else 1

    # Get dimensions
    if batch_first:
        # Convert from [batch, seq, input_size] to [seq, batch, input_size]
        input = op.Transpose(input, perm=[1, 0, 2])

    hidden_size = op.Shape(initial_h, start=2, end=3)

    # Process each layer
    current_input = input
    output_h_list = []
    output_c_list = []

    for layer_idx in range(num_layers):
        # Extract hidden and cell states for this layer
        layer_start = layer_idx * num_directions
        layer_end = (layer_idx + 1) * num_directions
        layer_h = op.Slice(initial_h, [layer_start], [layer_end], [0])
        layer_c = op.Slice(initial_c, [layer_start], [layer_end], [0])

        # Extract parameters for this layer
        # Parameter layout: [W_ih, W_hh, b_ih, b_hh] for each direction
        params_per_direction = 4 if has_biases else 2
        params_per_layer = params_per_direction * num_directions
        param_start_idx = layer_idx * params_per_layer

        # Build weight matrices for ONNX LSTM
        # ONNX expects: W[iofc] shape [num_directions, 4*hidden_size, input_size]
        # PyTorch provides: W_ih shape [4*hidden_size, input_size]
        W_list = []
        R_list = []
        B_list = [] if has_biases else None

        for dir_idx in range(num_directions):
            dir_param_start = param_start_idx + dir_idx * params_per_direction
            W_ih = params[
                dir_param_start
            ]  # [4*hidden_size, input_size] - PyTorch order: [i,f,g,o]
            W_hh = params[
                dir_param_start + 1
            ]  # [4*hidden_size, hidden_size] - PyTorch order: [i,f,g,o]

            # Reorder gates from PyTorch [i,f,g,o] to ONNX [i,o,f,g]
            # Split into individual gates
            W_ii = op.Slice(W_ih, [0], hidden_size, [0])
            W_if = op.Slice(W_ih, hidden_size, op.Mul(hidden_size, 2), [0])
            W_ig = op.Slice(
                W_ih,
                op.Mul(hidden_size, 2),
                op.Mul(hidden_size, 3),
                [0],
            )
            W_io = op.Slice(
                W_ih,
                op.Mul(hidden_size, 3),
                op.Mul(hidden_size, 4),
                [0],
            )

            W_hi = op.Slice(W_hh, [0], hidden_size, [0])
            W_hf = op.Slice(W_hh, hidden_size, op.Mul(hidden_size, 2), [0])
            W_hg = op.Slice(
                W_hh,
                op.Mul(hidden_size, 2),
                op.Mul(hidden_size, 3),
                [0],
            )
            W_ho = op.Slice(
                W_hh,
                op.Mul(hidden_size, 3),
                op.Mul(hidden_size, 4),
                [0],
            )

            # Reorder: [i,o,f,g]
            W_ih_reordered = op.Concat(
                W_ii, W_io, W_if, W_ig, axis=0
            )  # [4*hidden_size, input_size] - ONNX order
            W_hh_reordered = op.Concat(
                W_hi, W_ho, W_hf, W_hg, axis=0
            )  # [4*hidden_size, hidden_size] - ONNX order

            # Add direction dimension
            W_ih_expanded = op.Unsqueeze(
                W_ih_reordered, [0]
            )  # [1, 4*hidden_size, input_size]
            W_hh_expanded = op.Unsqueeze(
                W_hh_reordered, [0]
            )  # [1, 4*hidden_size, hidden_size]

            W_list.append(W_ih_expanded)
            R_list.append(W_hh_expanded)

            if has_biases:
                b_ih = params[
                    dir_param_start + 2
                ]  # [4*hidden_size] - PyTorch order: [i,f,g,o]
                b_hh = params[
                    dir_param_start + 3
                ]  # [4*hidden_size] - PyTorch order: [i,f,g,o]

                # Reorder biases from PyTorch [i,f,g,o] to ONNX [i,o,f,g]
                b_ii = op.Slice(b_ih, [0], hidden_size, [0])
                b_if = op.Slice(
                    b_ih,
                    hidden_size,
                    op.Mul(hidden_size, 2),
                    [0],
                )
                b_ig = op.Slice(
                    b_ih,
                    op.Mul(hidden_size, 2),
                    op.Mul(hidden_size, 3),
                    [0],
                )
                b_io = op.Slice(
                    b_ih,
                    op.Mul(hidden_size, 3),
                    op.Mul(hidden_size, 4),
                    [0],
                )

                b_hi = op.Slice(b_hh, [0], hidden_size, [0])
                b_hf = op.Slice(
                    b_hh,
                    hidden_size,
                    op.Mul(hidden_size, 2),
                    [0],
                )
                b_hg = op.Slice(
                    b_hh,
                    op.Mul(hidden_size, 2),
                    op.Mul(hidden_size, 3),
                    [0],
                )
                b_ho = op.Slice(
                    b_hh,
                    op.Mul(hidden_size, 3),
                    op.Mul(hidden_size, 4),
                    [0],
                )

                # Reorder: [i,o,f,g]
                b_ih_reordered = op.Concat(
                    b_ii, b_io, b_if, b_ig, axis=0
                )  # [4*hidden_size] - ONNX order
                b_hh_reordered = op.Concat(
                    b_hi, b_ho, b_hf, b_hg, axis=0
                )  # [4*hidden_size] - ONNX order

                # ONNX expects biases concatenated: [Wb[iofc], Rb[iofc]]
                b_combined = op.Concat(
                    b_ih_reordered, b_hh_reordered, axis=0
                )  # [8*hidden_size]
                b_expanded = op.Unsqueeze(b_combined, [0])  # [1, 8*hidden_size]
                B_list.append(b_expanded)

        # Concatenate weights for all directions
        W = op.Concat(*W_list, axis=0) if len(W_list) > 1 else W_list[0]
        R = op.Concat(*R_list, axis=0) if len(R_list) > 1 else R_list[0]
        B = (
            op.Concat(*B_list, axis=0)
            if has_biases and len(B_list) > 1
            else (B_list[0] if has_biases else None)
        )

        # Call ONNX LSTM operator
        direction = "bidirectional" if bidirectional else "forward"

        # Extract hidden_size from initial_h shape: [num_layers * num_directions, batch, hidden_size]
        hidden_size_attr = initial_h.shape[2]

        if B is not None:
            Y, Y_h, Y_c = op.LSTM(
                current_input,
                W,
                R,
                B,
                None,
                layer_h,
                layer_c,
                direction=direction,
                hidden_size=hidden_size_attr,
                outputs=3,
            )
        else:
            Y, Y_h, Y_c = op.LSTM(
                current_input,
                W,
                R,
                None,
                None,
                layer_h,
                layer_c,
                direction=direction,
                hidden_size=hidden_size_attr,
                outputs=3,
            )

        current_input = _rnn_output_to_layer_output(
            op, Y, bidirectional, hidden_size_attr
        )

        # Apply dropout if not last layer and dropout > 0
        if layer_idx < num_layers - 1 and dropout > 0.0 and train:
            current_input = residual.aten_dropout(op, current_input, dropout, train)

        # Store final hidden and cell states
        output_h_list.append(Y_h)
        output_c_list.append(Y_c)

    # Concatenate all layer outputs
    final_h = (
        output_h_list[0]
        if len(output_h_list) == 1
        else op.Concat(*output_h_list, axis=0)
    )
    final_c = (
        output_c_list[0]
        if len(output_c_list) == 1
        else op.Concat(*output_c_list, axis=0)
    )

    # Handle batch_first for output
    if batch_first:
        # Convert from [seq, batch, features] to [batch, seq, features]
        current_input = op.Transpose(current_input, perm=[1, 0, 2])

    return current_input, final_h, final_c
