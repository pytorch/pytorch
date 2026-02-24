# Plan: Rewrite LSTM decomposition to use `torch.scan`

## Problem

`one_layer_lstm` in `torch/_decomp/decompositions.py:3638` iterates over the
sequence dimension with a Python `for` loop (`for inp in precomputed_input`).
This calls `Tensor.__iter__` → `unbind(0)`, which requires materializing a
concrete sequence length, forcing specialization and breaking `Dim.DYNAMIC`.

## Solution

Replace the `for` loop in `one_layer_lstm` with `torch.scan`, which handles
dynamic iteration counts symbolically.

## Changes

**File: `torch/_decomp/decompositions.py`**

Replace `one_layer_lstm` (lines 3638-3662). The new implementation:

1. Keeps the parameter extraction and `F.linear` precomputation unchanged.
2. Removes the `precomputed_input.flip(0)` — instead passes `reverse` to `scan`.
3. Defines a `combine_fn` that closes over `hh_weight`, `hh_bias`, `hr_weight`
   and calls `lstm_cell`. The carry is `(hx, cx)` and the output is `hx`
   (cloned to avoid aliasing).
4. Calls `scan(combine_fn, init=(hx, cx), xs=precomputed_input, dim=0, reverse=reverse)`.
5. Squeezes the final carry's leading dim (as before) and squeezes the stacked
   output's dim-1 (the `1` from `[seq, 1, batch, hidden]` → `[seq, batch, hidden]`).

Sketch:
```python
def one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    precomputed_input = F.linear(inp, ih_weight, ih_bias)

    def combine_fn(carry, x):
        hx, cx = carry
        hx, cx = lstm_cell(x, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=2)
        return (hx, cx), hx.clone()

    (hx, cx), out = scan(combine_fn, (hx, cx), precomputed_input, dim=0, reverse=reverse)

    out = out.squeeze(1)  # [seq, 1, batch, hidden] -> [seq, batch, hidden]
    return out, (hx.squeeze(1), cx.squeeze(1))
```

**Import:** Add `from torch._higher_order_ops.scan import scan` near the top of
the file (or alongside existing higher-order-op imports).

## What this does NOT change

- `lstm_cell` — unchanged.
- `one_layer_lstm_data` (packed sequence variant) — unchanged; this handles
  variable batch sizes per timestep which is fundamentally incompatible with
  `scan` (each step has different shapes).
- `mkldnn_one_layer_lstm` — unchanged (opaque native op).
- `_rnn_helper`, `lstm_impl`, `lstm_data_impl` — unchanged.
- `select_one_layer_lstm_function` — unchanged.

## Testing

Run the user's `test2.py` with `Dim.DYNAMIC` on the sequence dim to verify the
export succeeds without the constraint violation error.
