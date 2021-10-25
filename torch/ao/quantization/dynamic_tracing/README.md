# Quantization with dynamic tracing (final name TBD)

Note: this code is an early prototype and the API may change at any time.

Quantization with dynamic tracing (final name TBD) is a prototype of automated
quantization syntax transforms for PyTorch Eager mode. High level algorithm:

1. take a user model and an example input
2. trace the model with example input and record the subgraphs of seen quantizeable ops
3. define quantization syntax transforms over the seen subgraphs
4. during execution of user code, dynamically call into the subgraphs from (3) when necessary

## User API overview

```
from torch.ao.quantization._quantize_dynamic_tracing import prepare, convert

m = M(...)
mp = prepare(m, example_input)
# calibration (not shown)
mq = convert(mp)
```

## Code overview

### auto_trace.py

This file contains the logic for partial program capture. We override `__torch_function__`
and `torch.nn.Module.__call__` to define the interception points. During tracing, calibration
and inference, we dynamically execute quantization transforms from these interception
points. The following pseudocode illustrates how both `add_auto_observation` and
`add_auto_convert` call into quantization hooks:

```
def __torch_function__(cls, func, types, args, kwargs):
  ...

  # the framework provides `cur_module` of the current function
  # `quant_state` is the quantization state of the current module
  quant_state = cur_module._auto_quant_state

  # only some ops are quantizeable, the following line allows us to terminate
  # early for unquantizeable ones
  needs_op_hooks = quant_state.cur_op_needs_hooks(func)

  if needs_op_hooks:

    # this line will throw an exception if control flow over quantizeable ops
    # is detected
    qstate.validate_cur_op(func)

    # "before" hook
    args, kwargs = qstate.op_prepare_before_hook(func, args, kwargs, ...)

    # run original function
    output = super().__torch_function(func, types, args, kwargs)

    # "after" hook
    output = qstate.op_prepare_after_hook(func, output, args, kwargs, ...)

  else:
    output = super().__torch_function(func, types, args, kwargs)

  ...

  return output
```

In detail:

#### calibration

This happens in the `add_auto_observation` function.

1. For each non-leaf module in the model, we create a new `AutoQuantizationState`
module and attach it as a child. This contains the quantization state
(subgraphs and observers).
2. For each `__torch_function__` and `torch.nn.Module.__call__` override, call
quantization hooks if necessary. If `first_call` is true, this captures the
subgraphs. Otherwise, this performs observation.

#### inference

This happens in the `add_auto_convert` function.

1. For each `__torch_function__` and `torch.nn.Module.__call__` override, call
quantization hooks if necessary. This performs the quantization inference
syntax transforms.

### quantization_state.py

This file defines `AutoQuantizationState`. This is an object which
stores quantization state for its parent module. It contains the following state:

1. all captured quantization op subgraphs
2. all dynamically created observers and fake_quants

It also contains the following hooks:

1. module before and after hooks (used for dtype transitions)
2. function before and after hooks (used for dtype transitions and observation)
3. function replacement hooks (used for substiting quantized kernels)
