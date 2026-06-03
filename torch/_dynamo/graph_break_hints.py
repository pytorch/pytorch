USER_ERROR = [
    "Your code may result in an error when running in eager. "
    "Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. "
    'You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. '
]
CUSTOM_OP_FAKE_TENSOR_SUBCLASS_DATA_PTR = [
    "A Tensor subclass argument reached this custom op during fake-value propagation. "
    "In this case, the subclass's `__torch_dispatch__` may run the custom op implementation "
    "instead of the registered fake impl.",
    "This happens while compiling with fake tensors; the same custom op may still run "
    "correctly in eager with real tensors.",
    "Handle FakeTensor-wrapped subclass inputs in the custom op implementation, or change "
    "the subclass `__torch_dispatch__` to unwrap to operations that have fake implementations.",
]
DYNAMO_BUG = [
    "This is likely to be a Dynamo bug. Please report an issue to PyTorch.",
]
DIFFICULT = [
    "This graph break may be difficult to debug. Please report an issue to PyTorch for assistance.",
]
FUNDAMENTAL = [
    "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through "
    "your code. Consider finding a workaround.",
]
SUPPORTABLE = [
    "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you "
    "encounter this graph break often and it is causing performance issues.",
]
CAUSED_BY_EARLIER_GRAPH_BREAK = [
    "This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.",
]
INFERENCE_MODE = [
    "Avoid using `tensor.is_inference()` and `torch.is_inference_mode_enabled()` in your compile code. "
    "This is primarily used in conjunction with `torch.inference_mode`. Consider using `torch.no_grad` instead "
    "because `torch.no_grad` leads to same improvements as `inference_mode` when `torch.compile` is used.",
]
SPARSE_TENSOR = [
    "Sparse tensor operations are not yet fully supported in torch.compile with fullgraph=True. "
    "Consider using fullgraph=False to allow graph breaks, or move sparse tensor creation "
    "outside the compiled region.",
]
