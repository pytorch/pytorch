import torch
import functools
import pickle


class TensorID(object):
    def __setstate__(self, id):
        self.id = id


class IntList(object):
    def __setstate__(self, data):
        self.data = data


class LiteralTensor(object):
    def __setstate__(self, data):
        buffer = data[0].encode('utf-8')
        sizes = data[1].data

        num_elements = functools.reduce(lambda size, acc: size * acc, sizes)
        storage = torch.Storage.from_buffer(buffer=buffer, byte_order="little", count=num_elements, offset=0)
        self.tensor = torch.tensor(storage.tolist()).view(sizes)


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
