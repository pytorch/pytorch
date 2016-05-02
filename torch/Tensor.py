# This will be included in __init__.py

class RealTensor(C.RealTensorBase):
    def __str__(self):
        return "RealTensor"

    def __repr__(self):
        return str(self)
