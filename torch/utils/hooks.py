import weakref


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""
    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        key = id(self)
        if hooks_dict is not None and key in hooks_dict:
            del hooks_dict[key]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()


def partial_apply_hook(hook, module):
    """Computes the partial application hook(module)

    Given a hook with the signature::

        hook(module, grad_input, grad_output) -> Tensor

    This binds the first argument `module` and returns a new function with the
    signature::

        wrapper(grad_input, grad_output) -> Tensor
    """
    def wrapper(grad_input, grad_output):
        return hook(module, grad_input, grad_output)
    # preserve the name for debugging
    wrapper.__name__ = hook.__name__
    return wrapper
