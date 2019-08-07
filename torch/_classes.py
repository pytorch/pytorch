import types

class _Classes(types.ModuleType):
    def __init__(self):
        super(_Classes, self).__init__('torch.classes')


# The classes "namespace"
classes = _Classes()
