# TODO: refactor nested_dict into common library with ATen
class nested_dict(object):
    def __init__(self, base, parent):
        self.base, self.parent = base, parent

    def __contains__(self, item):
        return item in self.base or item in self.parent

    def __getitem__(self, x):
        r = self.base.get(x)
        if r is not None:
            return r
        return self.parent[x]
