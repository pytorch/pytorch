
def _type(self, new_type=None, async=False):
    if new_type is None:
        return self.__module__ + '.' + self.__class__.__name__

    if isinstance(new_type, str):
        new_type = _import_dotted_name(new_type)
    if new_type == type(self):
        return self
    return new_type(self.size()).copy_(self, async)


def _range(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)


def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj

