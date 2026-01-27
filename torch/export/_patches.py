import contextlib
from collections.abc import Generator

import torch
from torch._decomp import global_decomposition_table
from torch._decomp.decompositions import _rnn_helper, gather_params, gru_cell, lstm_cell
from torch._higher_order_ops.while_loop import while_loop


def one_layer_while_loop_lstm(inp, hidden, params, has_biases, reverse=False):
    """
    1 layer fn for while loop LSTM

    Args:
        inp: Input tensor of shape (seq_len, batch, input_size)
        hidden: Tuple of (hx, cx) hidden states
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        reverse: Whether to process sequence in reverse

    Returns:
        Tuple of (output, (final_hx, final_cx))
    """
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    precomputed_input = torch.nn.functional.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input

    # while loop rewrite
    step_output = torch.empty(
        precomputed_input.size(0),
        *tuple(hx.shape[1:]),
        dtype=hx.dtype,
        device=hx.device,
    )

    def cond_fn(i, out, hx, cx):
        return i < precomputed_input.size(0)

    def body_fn(idx, out, hx, cx):
        # Extract the integer value from idx and constrain it for data-dependent indexing
        i = idx.item()
        torch._check_is_size(i)
        torch._check_is_size(i, max=precomputed_input.size(0) - 1)
        hx, cx = lstm_cell(
            precomputed_input[i], hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=2
        )
        out = out.clone()
        # Squeeze the first dimension before storing (lstm_cell preserves the unsqueezed dim)
        out[i] = hx.squeeze(0)
        return idx + 1, out, hx, cx

    cnt = torch.tensor(0, dtype=torch.int64)
    _, out, final_hx, final_cx = while_loop(
        cond_fn, body_fn, [cnt, step_output, hx, cx]
    )
    if reverse:
        out = out.flip(0)

    # Use squeeze(1) to match original implementation
    return out, (final_hx.squeeze(1), final_cx.squeeze(1))


def lstm_while_loop_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """
    LSTM implementation using while_loop for export compatibility.

    This is a drop-in replacement for the default LSTM decomposition that uses
    while_loop instead of Python loops, making it more suitable for torch.export.

    Args:
        input: Input tensor
        hx: Tuple of (h0, c0) hidden states
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        train: Training mode
        bidirectional: Whether to use bidirectional LSTM
        batch_first: Whether batch dimension is first

    Returns:
        Tuple of (output, h_n, c_n)
    """
    if len(hx) != 2:
        raise AssertionError("lstm expects two hidden states")
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    hidden = list(zip(hx[0], hx[1]))
    layer_fn = one_layer_while_loop_lstm
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    final_hiddens = list(zip(*final_hiddens))
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)


def one_layer_while_loop_gru(inp, hidden, params, has_biases, reverse=False):
    """
    1 layer fn for while loop GRU

    Args:
        inp: Input tensor of shape (seq_len, batch, input_size)
        hidden: Hidden state tensor
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        reverse: Whether to process sequence in reverse

    Returns:
        Tuple of (output, final_hidden)
    """
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    precomputed_input = torch.nn.functional.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    cur_hidden = hidden.unsqueeze(0)

    # while loop rewrite
    step_output = torch.empty(
        precomputed_input.size(0),
        *tuple(cur_hidden.shape[1:]),
        dtype=cur_hidden.dtype,
        device=cur_hidden.device,
    )

    def cond_fn(i, out, cur_hidden):
        return i < precomputed_input.size(0)

    def body_fn(idx, out, cur_hidden):
        # Extract the integer value from idx and constrain it for data-dependent indexing
        i = idx.item()
        torch._check_is_size(i)
        torch._check_is_size(i, max=precomputed_input.size(0) - 1)
        cur_hidden = gru_cell(
            precomputed_input[i], cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias
        )
        out = out.clone()
        out[i] = cur_hidden.squeeze(0)
        return idx + 1, out, cur_hidden

    cnt = torch.tensor(0, dtype=torch.int64)
    _, out, final_hidden = while_loop(cond_fn, body_fn, [cnt, step_output, cur_hidden])
    if reverse:
        out = out.flip(0)

    return out, final_hidden.squeeze(0)


def gru_while_loop_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """
    GRU implementation using while_loop for export compatibility.

    This is a drop-in replacement for the default GRU decomposition that uses
    while_loop instead of Python loops, making it more suitable for torch.export.

    Args:
        input: Input tensor
        hx: Hidden state tensor
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        num_layers: Number of GRU layers
        dropout: Dropout probability
        train: Training mode
        bidirectional: Whether to use bidirectional GRU
        batch_first: Whether batch dimension is first

    Returns:
        Tuple of (output, h_n)
    """
    params = gather_params(params, has_biases, False)
    hidden = list(hx.unbind(0))
    layer_fn = one_layer_while_loop_gru
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    return out, torch.stack(final_hiddens, 0)


@contextlib.contextmanager
def _register_rnn_while_loop_decomposition(
    rnn_op, rnn_impl
) -> Generator[None, None, None]:
    """
    Generic context manager for registering while_loop-based RNN decompositions.

    Args:
        rnn_op: The aten operation to patch (e.g., torch.ops.aten.lstm.input)
        rnn_impl: The while_loop-based implementation function

    Note:
        This is an internal helper. Use register_lstm_while_loop_decomposition()
        or register_gru_while_loop_decomposition() instead.
    """
    registry = global_decomposition_table["post_autograd"]

    # Save the original decomposition if it exists
    original_decomp = registry.get(rnn_op, None)

    # Save the original py_kernel if it exists
    original_py_kernel = rnn_op.py_kernels.get(
        torch._C.DispatchKey.CompositeImplicitAutograd, None
    )

    try:
        # Register our while_loop-based implementation
        registry[rnn_op] = rnn_impl
        rnn_op.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd] = rnn_impl
        yield
    finally:
        # Restore the original decomposition
        if original_decomp is not None:
            registry[rnn_op] = original_decomp
        else:
            # If there was no original, remove our registration
            registry.pop(rnn_op, None)

        # Restore the original py_kernel
        if original_py_kernel is not None:
            rnn_op.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd] = (
                original_py_kernel
            )
        else:
            # If there was no original, remove our registration
            rnn_op.py_kernels.pop(torch._C.DispatchKey.CompositeImplicitAutograd, None)


@contextlib.contextmanager
def register_lstm_while_loop_decomposition() -> Generator[None, None, None]:
    """
    Context manager that temporarily registers the while_loop-based LSTM decomposition.

    The while_loop-based decomposition is more suitable for export and graph-based
    execution, as it avoids Python control flow that cannot be captured in the graph.
    This should support dynamic sequence lengths, however as while_loop does not
    support Autograd yet, an ExportedProgram created with this will not be trainable.

    Usage::

        from torch.export._patches import register_lstm_while_loop_decomposition
        from torch.export import export

        with register_lstm_while_loop_decomposition():
            # Export your model with LSTM
            ep = export(model, (x, h0, c0))

    Note:
        This context manager temporarily modifies the global decomposition table
        and py_kernels registration. The original registrations are restored when
        exiting the context.
    """
    with _register_rnn_while_loop_decomposition(
        torch.ops.aten.lstm.input, lstm_while_loop_impl
    ):
        yield


@contextlib.contextmanager
def register_gru_while_loop_decomposition() -> Generator[None, None, None]:
    """
    Context manager that temporarily registers the while_loop-based GRU decomposition.

    The while_loop-based decomposition is more suitable for export and graph-based
    execution, as it avoids Python control flow that cannot be captured in the graph.
    This should support dynamic sequence lengths, however as while_loop does not
    support Autograd yet, an ExportedProgram created with this will not be trainable.

    Usage::

        from torch.export._patches import register_gru_while_loop_decomposition
        from torch.export import export

        with register_gru_while_loop_decomposition():
            # Export your model with GRU
            ep = export(model, (x, h0))

    Note:
        This context manager temporarily modifies the global decomposition table
        and py_kernels registration. The original registrations are restored when
        exiting the context.
    """
    with _register_rnn_while_loop_decomposition(
        torch.ops.aten.gru.input, gru_while_loop_impl
    ):
        yield
