import weakref
from collections.abc import MutableMapping
from typing import Dict


__all__ = ['WeakTensorRefKey', 'WeakTensorKeyDictionary']


# Utility classes for working with weak references to tensors

# torch.Tensors cannot be used as a key in a dictionary
# because they define a custom __eq__ function which when used
# to resolve hash collisions will throw when comparing tensors:
# "RuntimeError: bool value of Tensor with more than one value is ambiguous."
# To avoid that, we use an object which will hold a Tensor and use
# its id for both hashing and equality.
# In order to use this as a weak key reference, we cannot
# simply use weakref.WeakKeyDictionary because the newly constructed
# WeakTensorRefKey only use would be a dictionary so it would have no strong
# references.
# To get around this issue, we can use it as a normal key, and then set
# `weakref.finalize` to delete the key when its contained tensor dies.


class WeakTensorRefKey(object):
    def __init__(self, ten):
        self.ten = weakref.ref(ten)
        # store id since as soon as ten is deallocated
        # the old id will no longer be recoverable, and
        # we need to be able to remove the WeakTensorRefKey
        # from the dictionary by hashing it to the same
        # value it had when ten was alive
        self.id = id(self.ten())

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.id == other.id

class WeakTensorKeyDictionary(MutableMapping):
    data: Dict[WeakTensorRefKey, object]

    def __init__(self):
        self.data = {}

    def __contains__(self, k):
        return WeakTensorRefKey(k) in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        def generator():
            for wk in self.data:
                k = wk.ten()
                if k is not None:
                    yield k
        return generator()

    def __getitem__(self, k):
        return self.data[WeakTensorRefKey(k)]

    def __setitem__(self, k, v):
        wk = WeakTensorRefKey(k)
        self_weak_ref = weakref.ref(self)

        def del_ten():
            self_ref = self_weak_ref()
            if self_ref is None:
                return
            self_ref.data.pop(wk, None)
        weakref.finalize(k, del_ten)
        self.data[wk] = v

    def __delitem__(self, k):
        del self.data[WeakTensorRefKey(k)]
