
def _type(self, t=None):
    if isinstance(t, str) or t is None:
        current = self.__module__ + '.' + self.__class__.__name__
        if t is None:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        return _import_dotted_name(t)(self.size()).copy_(self)
    else:
        if t == type(self):
            return self
        return t(self.size()).copy_(self)


def _range(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)


def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj

