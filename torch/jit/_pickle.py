import pickle


class TensorID(object):
    def __setstate__(self, id):
        self.id = id


class IntList(object):
    def __setstate__(self, data):
        self.data = data


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if not module == '__main__':
            return None

        if name == 'TensorID':
            return TensorID
        elif name == 'IntList':
            return IntList
        elif name == 'LiteralTensor':
            return LiteralTensor
