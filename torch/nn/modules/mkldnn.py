from .module import Module

class Mkldnn(Module):
    def __init__(self):
        super(Mkldnn, self).__init__()

    def forward(self, input):
        return input.to_mkldnn()

class Dense(Module):
    def __init__(self):
        super(Dense, self).__init__()

    def forward(self, input):
        return input.to_dense()
