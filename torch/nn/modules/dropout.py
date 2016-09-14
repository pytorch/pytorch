from .module import Module

class Dropout(Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout(self.p, self.train, self.inplace)(input)


class Dropout2d(Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout2d(self.p, self.train, self.inplace)(input)


class Dropout3d(Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout3d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout3d(self.p, self.train, self.inplace)(input)

