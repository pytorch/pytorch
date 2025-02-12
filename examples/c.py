import torch
import inspect
import functools

z = torch.ones(3, 3) * 4


def fn(x, y=torch.ones(3, 3) * 2):
    global z
    return torch.cos(y) + x + z


def export(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    dynamo_gm = None
    graphargs = None

    def backend(gm, *args):
        nonlocal dynamo_gm, graphargs
        # TODO - Find a more blessed way to do this.

        # graphargs has the source for each dynamo graph input
        graphargs = torch._dynamo.output_graph.OUTPUT_GRAPH.graphargs

        dynamo_gm = gm
        return gm.forward

    torch.compile(fn, backend=backend, fullgraph=True)(*args, **kwargs)

    # This must be called after torch.compile has run because Dynamo adds new kv
    # pairs to fn globals.
    scope = {
        "G": fn.__globals__,
    }

    def exported_fn(*r_args, **r_kwargs):
        nonlocal graphargs, scope, dynamo_gm

        # Overhead includes
        # 1) inspect.bind + apply_defaults
        # 2) Updating dict
        # 3) eval - this might be fine though because we need this anyways to access the parameters etc.
        bound = signature.bind(*r_args, **r_kwargs)
        bound.apply_defaults()
        local_scope = bound.arguments
        scope["L"] = local_scope

        dynamo_args = []
        for grapharg in graphargs:
            arg_name = grapharg.source.name()
            val = eval(arg_name, scope)
            dynamo_args.append(val)

        out = dynamo_gm(*dynamo_args)

        # Ensure that there are no extra references leaking out
        scope["L"] = None

        # The output signature matching will have to use the old one.
        return out[0]

    return exported_fn


x = torch.randn(3, 3)
y = torch.randn(3, 3)
e = export(fn, torch.randn(3, 3), torch.randn(3, 3))

print(fn(x))
print(e(x))
