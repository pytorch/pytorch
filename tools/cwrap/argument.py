from copy import deepcopy

class Argument(object):
    def __init__(self, type, name):
        self.type = type
        self.name = name

    def __hash__(self):
        return (self.type + '#' + self.name).__hash__()

    def copy(self):
        return deepcopy(self)
