
_magic_methods = ['__subclasscheck__', '__hex__', '__rmul__',
                  '__float__', '__idiv__', '__setattr__', '__div__', '__invert__',
                  '__nonzero__', '__rshift__',
                  '__eq__', '__pos__', '__round__',
                  '__rand__', '__or__', '__complex__', '__divmod__',
                  '__len__', '__reversed__', '__copy__', '__reduce__',
                  '__deepcopy__', '__rdivmod__', '__rrshift__', '__ifloordiv__',
                  '__hash__', '__iand__', '__xor__', '__isub__', '__oct__',
                  '__ceil__', '__imod__', '__add__', '__truediv__',
                  '__unicode__', '__le__', '__delitem__', '__sizeof__', '__sub__',
                  '__ne__', '__pow__', '__bytes__', '__mul__',
                  '__itruediv__', '__bool__', '__iter__', '__abs__',
                  '__gt__', '__iadd__', '__enter__',
                  '__floordiv__', '__call__', '__neg__',
                  '__and__', '__ixor__', '__getitem__', '__exit__', '__cmp__',
                  '__getstate__', '__index__', '__contains__', '__floor__', '__lt__', '__getattr__',
                  '__mod__', '__trunc__', '__delattr__', '__instancecheck__', '__setitem__', '__ipow__',
                  '__ilshift__', '__long__', '__irshift__', '__imul__',
                  '__lshift__', '__dir__', '__ge__', '__int__', '__ior__']


class MockedObject:
    _name: str

    def __init__(self, name):
        self.__dict__['_name'] = name

    def __repr__(self):
        return f"MockedObject({self._name})"


def install_method(method_name):
    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError(f"Object '{self._name}' was mocked out during packaging but it is being used in {method_name}")
    setattr(MockedObject, method_name, _not_implemented)

for method_name in _magic_methods:
    install_method(method_name)
