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
