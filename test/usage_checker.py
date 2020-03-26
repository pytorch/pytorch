# Import this module to introduce memory format checks into eager 
# execution mode.
# 
# After importing, operators will raise an exception if the output of 
# the operator doesn't match the memory format of the input.
#
# Note: works only for torch.channels_last memory format
#
import torch


def contains_tensors_4d_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last) and t.shape[1] > 1 and (t.shape[2] > 1 or t.shape[3] > 1):
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_tensors_4d_cl(list(t)):
                return True
    return False


def contains_tensors_4d(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.dim() == 4:
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_tensors_4d(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    res = ''
    for t in args:
        if isinstance(t, torch.Tensor):
            res += "%s %s %s %s %s\n" % (indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            res += "%s %s\n" % (indent, type(t))
            res += print_inputs(list(t), indent=indent + '    ') + "\n"
        else:
            res += "%s %s\n" % (indent, t)
    return res


def check_wrapper(fn):
    name = fn.__name__
    def check_cl(*args, **kwargs):
        have_4d_tensor = contains_tensors_4d(args)
        if have_4d_tensor:
            res = "----------------------------------\n%s\n%s" % (name, print_inputs(args))
            with open("USAGE", "a+") as f:
                f.write(res)
        result = fn(*args, **kwargs)
        supports = contains_tensors_4d_cl(args) and contains_tensors_4d_cl([result])
        if supports:
            res = "----------------------------------\noperator name:%s\n%s" % (name, print_inputs(args))
            with open("SUPPORTS", "a+") as f:
                f.write(res)
                f.write("\n  outputs:\n")
                f.write(print_inputs([result]))
        return result
    return check_cl


def attribute(m):
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['backward', 'autograd', 'is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__', 'dim', 'tensor']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


attribute(torch.Tensor)
attribute(torch.nn.functional)
attribute(torch)
