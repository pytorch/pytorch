"""Gradient interface"""

def _get_not_implemented_function(func_name):
    def tmp(*args, **kwargs):
        err_message = "{} has been removed as it is not representative of the actual backward.\n"
        err_message += "If you need the old implementation of this function, you can copy/paste "
        err_message += "it from https://github.com/pytorch/pytorch/blob/v1.3.1/torch/nn/grad.py"
        raise RuntimeError(err_message.format(func_name))
    return tmp

conv1d_input = _get_not_implemented_function("conv1d_input")
conv1d_weight = _get_not_implemented_function("conv1d_weight")
conv2d_input = _get_not_implemented_function("conv2d_input")
conv2d_weight = _get_not_implemented_function("conv2d_weight")
conv3d_input = _get_not_implemented_function("conv3d_input")
conv3d_weight = _get_not_implemented_function("conv3d_weight")
