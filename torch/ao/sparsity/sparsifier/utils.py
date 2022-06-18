from torch import nn

__all__ = ["module_to_fqn", "fqn_to_module", "get_arg_info_from_tensor_fqn", "FakeSparsity"]

def module_to_fqn(model, module, prefix=''):
    for name, child in model.named_children():
        new_name = prefix + '.' + name
        if child is module:
            return new_name
        child_path = module_to_fqn(child, module, prefix=new_name)
        if child_path is not None:
            return child_path
    return None

def fqn_to_module(model, path):
    path = path.split('.')
    for name in path:
        model = getattr(model, name, None)
        if model is None:
            return None
    return model

def get_arg_info_from_tensor_fqn(model, tensor_fqn):
    # remove starting '.' from tensor_fqn if it exists
    if tensor_fqn[0] == '.':
        tensor_fqn = tensor_fqn[1:]

    # string manip to split tensor_fqn into module_fqn and tensor_name
    # if tensor_fqn is 'weight' then module_fqn and tensor_name are '' and 'weight'
    # if tensor_fqn is 'linear.weight' then module_fqn and tensor_name are 'linear' and 'weight'
    tensor_name = tensor_fqn.split('.')[-1]
    module_fqn = tensor_fqn[:-len(tensor_name) - ('.' in tensor_fqn)]


    module = fqn_to_module(model, module_fqn)
    if module is None:  # handling for module_fqn=''
        module = model

    return {
        'module_fqn': module_fqn,
        'module': module,
        'tensor_name': tensor_name,
        'tensor_fqn': tensor_fqn,
    }

# Parametrizations
class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parmeter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        assert self.mask.shape == x.shape
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return dict()
