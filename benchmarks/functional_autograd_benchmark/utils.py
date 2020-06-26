from collections import defaultdict

# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    orig_params = tuple(p.detach().requires_grad_() for p in orig_params)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def get_str(res, header=None):
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

def read_str(out):
    out = out.strip().split("\n")
    out = out[2:]  # Ignore the header lines

    res = defaultdict(defaultdict)

    for line in out:
        model, task, mean, var = [f.strip() for f in line.strip().split("|") if f]
        res[model][task] = (float(mean), float(var))

    return res
