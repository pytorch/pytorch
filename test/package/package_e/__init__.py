result = "package_a"


class PackageEObject:
    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj

    def return_result(self):
        return result
