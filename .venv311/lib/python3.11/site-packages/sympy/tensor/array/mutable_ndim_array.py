from sympy.tensor.array.ndim_array import NDimArray


class MutableNDimArray(NDimArray):

    def as_immutable(self):
        raise NotImplementedError("abstract method")

    def as_mutable(self):
        return self

    def _sympy_(self):
        return self.as_immutable()
