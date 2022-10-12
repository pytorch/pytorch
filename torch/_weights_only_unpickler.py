# Very restricted unpickler
# Based of https://github.com/python/cpython/blob/main/Lib/pickle.py
# Expected to be useful for loading PyTorch model weights
# For example:
# data = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth').read()
# buf = io.BytesIO(data)
# weights = torch.load(buf, pickle_module=WeightsUnpickler)

from sys import maxsize
from struct import unpack
from collections import OrderedDict

import torch
from pickle import (UnpicklingError, bytes_types, decode_long,
                    STOP, PROTO,
                    MARK,
                    # Risky ops: class resolution, state modification, function invokation
                    GLOBAL, BUILD, REDUCE, NEWOBJ, APPENDS,
                    # Construct tirivial objects
                    NONE, NEWTRUE, NEWFALSE, EMPTY_DICT, EMPTY_LIST, EMPTY_TUPLE, EMPTY_SET,
                    LONG1, LONG_BINGET,
                    BININT, BININT1, BININT2, BINPERSID, BINUNICODE,
                    BINGET, BINPUT, LONG_BINPUT, SETITEM, SETITEMS, TUPLE, TUPLE1, TUPLE2, TUPLE3)


# Unpickling machinery

class Unpickler:

    def __init__(self, file, *, encoding: str = "UNUSED"):
        self.readline = file.readline
        self.read = file.read
        self.memo = {}

    def load(self):
        """Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        """
        self.metastack = []
        self.stack = []
        self.append = self.stack.append
        read = self.read
        readline = self.readline
        while True:
            key = read(1)
            if not key:
                raise EOFError
            assert isinstance(key, bytes_types)
            if key[0] == STOP[0]:
                rc = self.stack.pop()
                return rc
            elif key[0] == PROTO[0]:
                # Read and ignore proto version
                read(1)[0]
                pass
            elif key[0] == NONE[0]:
                self.append(None)
            elif key[0] == GLOBAL[0]:
                module = readline()[:-1].decode("utf-8")
                name = readline()[:-1].decode("utf-8")
                full_path = f"{module}.{name}"
                ALLOWED_GLOBALS = {
                    "collections.OrderedDict": OrderedDict,
                    "torch.FloatTensor": torch.FloatTensor,
                    "torch.FloatStorage": torch.FloatStorage,
                    "torch.LongTensor": torch.LongTensor,
                    "torch.LongStorage": torch.FloatStorage,
                    "torch.nn.parameter.Parameter": torch.nn.Parameter,
                    "torch._utils._rebuild_parameter": torch._utils._rebuild_parameter,
                    "torch._utils._rebuild_tensor_v2": torch._utils._rebuild_tensor_v2,
                }
                if full_path in ALLOWED_GLOBALS:
                    self.append(ALLOWED_GLOBALS[full_path])
                else:
                    raise RuntimeError(f"Unsupported class {full_path}")
            elif key[0] == NEWOBJ[0]:
                args = self.stack.pop()
                cls = self.stack.pop()
                if cls != torch.nn.Parameter:
                    raise RuntimeError("Trying to instantiate unsupported class")
                self.append(cls.__new__(cls, *args))
            elif key[0] == REDUCE[0]:
                args = self.stack.pop()
                func = self.stack[-1]
                self.stack[-1] = func(*args)
            elif key[0] == BUILD[0]:
                state = self.stack.pop()
                inst = self.stack[-1]
                if type(inst) is torch.nn.Parameter:
                    inst.__setstate__(state)
                elif type(inst) is OrderedDict:
                    inst.__dict__.update(state)
                else:
                    raise RuntimeError("Can only build parameter and dict objects")
            elif key[0] == APPENDS[0]:
                items = self.pop_mark()
                list_obj = self.stack[-1]
                if type(list_obj) is not list or not hasattr(list_obj, "extend"):
                    raise RuntimeError("Can only extend lists")
                list_obj.extend(items)
            elif key[0] == NEWFALSE[0]:
                self.append(False)
            elif key[0] == NEWTRUE[0]:
                self.append(True)
            elif key[0] == EMPTY_TUPLE[0]:
                self.append(())
            elif key[0] == EMPTY_LIST[0]:
                self.append([])
            elif key[0] == EMPTY_DICT[0]:
                self.append({})
            elif key[0] == EMPTY_SET[0]:
                self.append(set())
            elif key[0] == BININT[0]:
                self.append(unpack('<i', read(4))[0])
            elif key[0] == BININT1[0]:
                self.append(self.read(1)[0])
            elif key[0] == BININT2[0]:
                self.append(unpack('<H', read(2))[0])
            elif key[0] == BINUNICODE[0]:
                strlen = unpack('<I', read(4))[0]
                if strlen > maxsize:
                    raise RuntimeError("String is too long")
                strval = str(read(strlen), 'utf-8', 'surrogatepass')
                self.append(strval)
            elif key[0] == BINPERSID[0]:
                pid = self.stack.pop()
                self.append(self.persistent_load(pid))
            elif key[0] in [BINGET[0], LONG_BINGET[0]]:
                idx = (read(1) if key[0] == BINGET[0] else unpack('<I', read(4)))[0]
                self.append(self.memo[idx])
            elif key[0] in [BINPUT[0], LONG_BINPUT[0]]:
                i = (read(1) if key[0] == BINPUT[0] else unpack('<I', read(4)))[0]
                if i < 0:
                    raise ValueError("negative argument")
                self.memo[i] = self.stack[-1]
            elif key[0] == LONG1[0]:
                n = read(1)[0]
                data = read(n)
                self.append(decode_long(data))
            elif key[0] == SETITEM[0]:
                (v, k) = (self.stack.pop(), self.stack.pop())
                self.stack[-1][k] = v
            elif key[0] == SETITEMS[0]:
                items = self.pop_mark()
                for i in range(0, len(items), 2):
                    self.stack[-1][items[i]] = items[i + 1]
            elif key[0] == MARK[0]:
                self.metastack.append(self.stack)
                self.stack = []
                self.append = self.stack.append
            elif key[0] == TUPLE[0]:
                items = self.pop_mark()
                self.append(tuple(items))
            elif key[0] == TUPLE1[0]:
                self.stack[-1] = (self.stack[-1],)
            elif key[0] == TUPLE2[0]:
                self.stack[-2:] = [(self.stack[-2], self.stack[-1])]
            elif key[0] == TUPLE3[0]:
                self.stack[-3:] = [(self.stack[-3], self.stack[-2], self.stack[-1])]
            else:
                raise RuntimeError(f"Unsupported operatnd {key[0]}")

    # Return a list of items pushed in the stack after last MARK instruction.
    def pop_mark(self):
        items = self.stack
        self.stack = self.metastack.pop()
        self.append = self.stack.append
        return items

    def persistent_load(self, pid):
        raise UnpicklingError("unsupported persistent id encountered")

def load(file, *, encoding: str = "UNUSED"):
    return Unpickler(file).load()
