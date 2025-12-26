import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def fn(x):
    y = x.nonzero()
    torch._check(y.shape[0] > 2) # dropped post dynamo (eg. gone by aot_inference_graph)
    return torch.randn(y.shape[0])

def true_fn(x):
    return fn(x)

def false_fn(x):
    return fn(x)

@torch.compile(dynamic=True)
def foo(x):
    x = x.sin().sin()

    b = torch.cond(
        pred=(x.sum() > 0),
        true_fn=lambda: true_fn(x),
        false_fn=lambda: false_fn(x)
    )

    torch._check(b.shape[0] > 2) # also redundant and not DCE'd
    return b

foo(torch.randn(2, 2, device="cuda"))
