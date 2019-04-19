import pickle

# This flag has torch.load skip looking for a magic number, protocol version,
# and sys info
_unpickle_skip_metadata = True


class TensorID(object):
    def __setstate__(self, id):
        self.id = id


class IntList(object):
    def __setstate__(self, data):
        self.data = data


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TensorID':
            return TensorID
        elif name == 'IntList':
            return IntList

        return super(Unpickler, self).find_class(module, name)


def load(*args):
    return pickle.load(*args)
