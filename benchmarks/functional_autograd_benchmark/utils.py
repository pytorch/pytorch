from collections import defaultdict

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj, names):
    """
    Deletes the attribute specified by the given list of names.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj, names, val):
    """
    Set the attribute specified by the given list of names to val.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], val)

def extract_weights(mod):
    """
    This function removes all the Parameters from the model and
    return them as a tuple.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    orig_params = tuple(p.detach().requires_grad_() for p in orig_params)
    return orig_params, names

def load_weights(mod, names, params):
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)

# Utilities to read/write markdown table-like content.
def to_markdown_table(res, header=None):
    if header is None:
        header = ("model", "task", "mean", "var")
    out = ""

    def write_line(*args):
        nonlocal out
        out += "| {} |\n".format(" | ".join(str(a) for a in args))

    # Make it a markdown table
    write_line(*header)
    write_line(*["--"] * len(header))
    for model, tasks in res.items():
        for task, line in tasks.items():
            write_line(*(model, task) + line)

    return out

def from_markdown_table(out):
    out = out.strip().split("\n")
    out = out[2:]  # Ignore the header lines

    res = defaultdict(defaultdict)

    for line in out:
        model, task, mean, var = [f.strip() for f in line.strip().split("|") if f]
        res[model][task] = (float(mean), float(var))

    return res
