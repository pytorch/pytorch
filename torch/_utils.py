import torch


def _type(self, t=None):
    if isinstance(t, str) or t is None:
        current = self.__module__ + '.' + self.__class__.__name__
        if t is None:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        return torch._import_dotted_name(t)(self.size()).copy_(self)
    else:
        if t == type(self):
            return self
        return t(self.size()).copy_(self)

