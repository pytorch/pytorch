from torch import nn

def module_to_fqn(model, layer, prefix=''):
    for name, child in model.named_children():
        new_name = prefix + '.' + name
        if child is layer:
            return new_name
        child_path = module_to_fqn(child, layer, prefix=new_name)
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
