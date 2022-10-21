# Unpickler restricted to loading only state dicts
# Restrict constructing types to a list defined in _get_allowed_globals()
# Restrict BUILD operation to `Tensor`, `Parameter` and `OrderedDict` types only
# Restrict APPEND/APPENDS to `list`
# In `GLOBALS` operation do not do class lookup by name, but rather rely on dictionary
# defined by `_get_allowed_globals()` method, that contains:
# - torch types (Storage, dtypes, Tensor, `torch.Size`),
# - `torch._utils._rebuild` functions.
# - `torch.nn.Parameter`
# - `collections.OrderedDict`

# Based of https://github.com/python/cpython/blob/main/Lib/pickle.py
# Expected to be useful for loading PyTorch model weights
# For example:
# data = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth').read()
# buf = io.BytesIO(data)
# weights = torch.load(buf, weights_only = True)

import functools as _functools
from collections import OrderedDict
from pickle import (
    APPEND,
    APPENDS,
    BINGET,
    BININT,
    BININT1,
    BININT2,
    BINPERSID,
    BINPUT,
    BINUNICODE,
    BUILD,
    bytes_types,
    decode_long,
    EMPTY_DICT,
    EMPTY_LIST,
    EMPTY_SET,
    EMPTY_TUPLE,
    GLOBAL,
    LONG1,
    LONG_BINGET,
    LONG_BINPUT,
    MARK,
    NEWFALSE,
    NEWOBJ,
    NEWTRUE,
    NONE,
    PROTO,
    REDUCE,
    SETITEM,
    SETITEMS,
    SHORT_BINSTRING,
    STOP,
    TUPLE,
    TUPLE1,
    TUPLE2,
    TUPLE3,
    UnpicklingError,
)
from struct import unpack
from sys import maxsize
from typing import Any, Dict, List

import torch


# Unpickling machinery
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    rc: Dict[str, Any] = {
        "collections.OrderedDict": OrderedDict,
        "torch.nn.parameter.Parameter": torch.nn.Parameter,
        "torch.serialization._get_layout": torch.serialization._get_layout,
        "torch.Size": torch.Size,
        "torch.Tensor": torch.Tensor,
    }
    # dtype
    for t in [
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        rc[str(t)] = t
    # Tensor classes
    for tt in torch._tensor_classes:
        rc[f"{tt.__module__}.{tt.__name__}"] = tt
    # Storage classes
    for ts in torch._storage_classes:
        rc[f"{ts.__module__}.{ts.__name__}"] = ts
    # Rebuild functions
    for f in [
        torch._utils._rebuild_parameter,
        torch._utils._rebuild_tensor,
        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_sparse_tensor,
        torch._utils._rebuild_meta_tensor_no_storage,
        torch._utils._rebuild_sparse_csr_tensor,
    ]:
        rc[f"torch._utils.{f.__name__}"] = f
    return rc


class Unpickler:
    def __init__(self, file, *, encoding: str = "bytes"):
        self.encoding = encoding
        self.readline = file.readline
        self.read = file.read
        self.memo: Dict[int, Any] = {}

    def load(self):
        """Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        """
        self.metastack = []
        self.stack: List[Any] = []
        self.append = self.stack.append
        read = self.read
        readline = self.readline
        while True:
            key = read(1)
            if not key:
                raise EOFError
            assert isinstance(key, bytes_types)
            # Risky operators
            if key[0] == GLOBAL[0]:
                module = readline()[:-1].decode("utf-8")
                name = readline()[:-1].decode("utf-8")
                full_path = f"{module}.{name}"
                if full_path in _get_allowed_globals():
                    self.append(_get_allowed_globals()[full_path])
                else:
                    raise RuntimeError(f"Unsupported class {full_path}")
            elif key[0] == NEWOBJ[0]:
                args = self.stack.pop()
                cls = self.stack.pop()
                if cls is not torch.nn.Parameter:
                    raise RuntimeError(f"Trying to instantiate unsupported class {cls}")
                self.append(torch.nn.Parameter(*args))
            elif key[0] == REDUCE[0]:
                args = self.stack.pop()
                func = self.stack[-1]
                if func not in _get_allowed_globals().values():
                    raise RuntimeError(
                        f"Trying to call reduce for unrecognized function {func}"
                    )
                self.stack[-1] = func(*args)
            elif key[0] == BUILD[0]:
                state = self.stack.pop()
                inst = self.stack[-1]
                if type(inst) is torch.Tensor:
                    # Legacy unpickling
                    inst.set_(*state)
                elif type(inst) is torch.nn.Parameter:
                    inst.__setstate__(state)
                elif type(inst) is OrderedDict:
                    inst.__dict__.update(state)
                else:
                    raise RuntimeError(
                        f"Can only build Tensor, parameter or dict objects, but got {type(inst)}"
                    )
            # Stack manipulation
            elif key[0] == APPEND[0]:
                item = self.stack.pop()
                list_obj = self.stack[-1]
                if type(list_obj) is not list:
                    raise RuntimeError(
                        f"Can only append to lists, but got {type(list_obj)}"
                    )
                list_obj.append(item)
            elif key[0] == APPENDS[0]:
                items = self.pop_mark()
                list_obj = self.stack[-1]
                if type(list_obj) is not list:
                    raise RuntimeError(
                        f"Can only extend lists, but got {type(list_obj)}"
                    )
                list_obj.extend(items)
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
            # Basic types construction
            elif key[0] == NONE[0]:
                self.append(None)
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
                self.append(unpack("<i", read(4))[0])
            elif key[0] == BININT1[0]:
                self.append(self.read(1)[0])
            elif key[0] == BININT2[0]:
                self.append(unpack("<H", read(2))[0])
            elif key[0] == BINUNICODE[0]:
                strlen = unpack("<I", read(4))[0]
                if strlen > maxsize:
                    raise RuntimeError("String is too long")
                strval = str(read(strlen), "utf-8", "surrogatepass")
                self.append(strval)
            elif key[0] == SHORT_BINSTRING[0]:
                strlen = read(1)[0]
                strdata = read(strlen)
                if self.encoding != "bytes":
                    strdata = strdata.decode(self.encoding, "strict")
                self.append(strdata)
            elif key[0] == BINPERSID[0]:
                pid = self.stack.pop()
                # Only allow persistent load of storage
                if type(pid) is not tuple and not type(pid) is not int:
                    raise RuntimeError(
                        f"persistent_load id must be tuple or int, but got {type(pid)}"
                    )
                if (
                    type(pid) is tuple
                    and len(pid) > 0
                    and torch.serialization._maybe_decode_ascii(pid[0]) != "storage"
                ):
                    raise RuntimeError(
                        f"Only persistent_load of storage is allowed, but got {pid[0]}"
                    )
                self.append(self.persistent_load(pid))
            elif key[0] in [BINGET[0], LONG_BINGET[0]]:
                idx = (read(1) if key[0] == BINGET[0] else unpack("<I", read(4)))[0]
                self.append(self.memo[idx])
            elif key[0] in [BINPUT[0], LONG_BINPUT[0]]:
                i = (read(1) if key[0] == BINPUT[0] else unpack("<I", read(4)))[0]
                if i < 0:
                    raise ValueError("negative argument")
                self.memo[i] = self.stack[-1]
            elif key[0] == LONG1[0]:
                n = read(1)[0]
                data = read(n)
                self.append(decode_long(data))
            # First and last deserializer ops
            elif key[0] == PROTO[0]:
                # Read and ignore proto version
                read(1)[0]
            elif key[0] == STOP[0]:
                rc = self.stack.pop()
                return rc
            else:
                raise RuntimeError(f"Unsupported operand {key[0]}")

    # Return a list of items pushed in the stack after last MARK instruction.
    def pop_mark(self):
        items = self.stack
        self.stack = self.metastack.pop()
        self.append = self.stack.append
        return items

    def persistent_load(self, pid):
        raise UnpicklingError("unsupported persistent id encountered")


def load(file, *, encoding: str = "ASCII"):
    return Unpickler(file, encoding=encoding).load()
