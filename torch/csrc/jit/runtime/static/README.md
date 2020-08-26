> :warning: **This is an experimental feature**

# Static Runtime

The premise of this approach is that a small subset of neural networks are well represented by a 
completely flattened dataflow graph.
TorchScript supports a far more feature programming paradigm,
so many models will not work out of the box.

## Assumptions

This is a list of current assumptions for use with
this feature.

- Inference only execution
- Single CPU device

After `torch.jit.freeze` and inlining/constant propagation is run on the model:

- No control flow
- No submodule invocations
- No references to `self`
- Inlined weights (i.e. no calls to `GetAttr`)

## Planned features

- Memory planning
- Operator dispatch inlining
- Operator subsitution
- Weight layout transformations (pre-packing)
- Lowering to `torch.jit.tensorexpr`
