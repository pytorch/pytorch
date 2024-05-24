# Unpickler restricted to loading only state dicts
# Restrict constructing types to a list defined in _get_allowed_globals()
# Restrict BUILD operation to `Tensor`, `Parameter` and `OrderedDict` types only
# Restrict APPEND/APPENDS to `list`
# In `GLOBALS` operation do not do class lookup by name, but rather rely on dictionary
# defined by `_get_allowed_globals()` method, that contains:
# - torch types (Storage, dtypes, Tensor, `torch.Size`),
# - `torch._utils._rebuild` functions.
# - `torch.nn.Parameter`
# - `collections.Counter`
# - `collections.OrderedDict`
# Additionally, users can use an allowlist for adding classes they have deemed as safe using
# `_add_safe_globals()` (`torch.serialization.add_safe_globals`)
# `_clear_safe_globals()` (`torch.serialization.clear_safe_globals`)
# `_get_safe_globals()` (`torch.serialization.get_safe_globals`)

# Based of https://github.com/python/cpython/blob/main/Lib/pickle.py
# Expected to be useful for loading PyTorch model weights
# For example:
# data = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth').read()
# buf = io.BytesIO(data)
# weights = torch.load(buf, weights_only = True)

import functools as _functools
from collections import Counter, OrderedDict
from inspect import getattr_static
from pickle import (
    APPEND,
    APPENDS,
    BINFLOAT,
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
from sys import maxsize, modules
from typing import Any, Dict, List, Type

import torch

_marked_safe_globals_list: List[Any] = []


def _add_safe_globals(safe_globals: List[Any]):
    global _marked_safe_globals_list
    _marked_safe_globals_list += safe_globals


def _get_safe_globals() -> List[Any]:
    global _marked_safe_globals_list
    return _marked_safe_globals_list


def _clear_safe_globals():
    global _marked_safe_globals_list
    _marked_safe_globals_list = []


# Separate from _get_allowed_globals because of the lru_cache on _get_allowed_globals
# For example if user had a script like
#   torch.load(file_a)
#   torch.serialization._add_safe_globals([torch.foo])
#   torch.load(file_b)
# the dynamic additions to safe_globals would not be picked up by
# _get_allowed_globals due to the lru_cache
def _get_user_allowed_globals():
    rc: Dict[str, Any] = {}
    for f in _marked_safe_globals_list:
        rc[f"{f.__module__}.{f.__name__}"] = f
    return rc


def _tensor_rebuild_functions():
    return {
        torch._utils._rebuild_parameter,
        torch._utils._rebuild_parameter_with_state,
        torch._utils._rebuild_qtensor,
        torch._utils._rebuild_tensor,
        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_tensor_v3,
        torch._utils._rebuild_sparse_tensor,
        torch._utils._rebuild_meta_tensor_no_storage,
        torch._utils._rebuild_nested_tensor,
        torch._utils._rebuild_wrapper_subclass,
    }


# Unpickling machinery
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    rc: Dict[str, Any] = {
        "collections.OrderedDict": OrderedDict,
        "collections.Counter": Counter,
        "torch.nn.parameter.Parameter": torch.nn.Parameter,
        "torch.serialization._get_layout": torch.serialization._get_layout,
        "torch.Size": torch.Size,
        "torch.Tensor": torch.Tensor,
        "torch.device": torch.device,
    }
    # dtype
    for t in torch.storage._dtype_to_storage_type_map().keys():
        rc[str(t)] = t
    for t in torch.storage._new_dtypes():
        rc[str(t)] = t
    # Tensor classes
    for tt in torch._tensor_classes:
        rc[f"{tt.__module__}.{tt.__name__}"] = tt
    # Storage classes
    for ts in torch._storage_classes:
        if ts not in (torch.storage.TypedStorage, torch.storage.UntypedStorage):
            # Wrap legacy storage types in a dummy class
            rc[f"{ts.__module__}.{ts.__name__}"] = torch.serialization.StorageType(
                ts.__name__
            )
        else:
            rc[f"{ts.__module__}.{ts.__name__}"] = ts
    # Quantization specific
    for qt in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
        torch.per_channel_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine_float_qparams,
    ]:
        rc[str(qt)] = qt
    # Rebuild functions
    for f in _tensor_rebuild_functions():
        rc[f"torch._utils.{f.__name__}"] = f

    # Handles Tensor Subclasses, Tensor's with attributes.
    # NOTE: It calls into above rebuild functions for regular Tensor types.
    rc["torch._tensor._rebuild_from_type_v2"] = torch._tensor._rebuild_from_type_v2
    return rc


class Unpickler:
    def __init__(self, file, *, encoding: str = "bytes"):
        self.encoding = encoding
        self.readline = file.readline
        self.read = file.read
        self.memo: Dict[int, Any] = {}
        # tensor subclass types found from GLOBAL instructions that have passed the criteria
        # to be allowed as the second argument to `torch._tensor._rebuild_from_type_v2`
        # This enables rebuilding of tensor subclasses defined outside the `torch` package.
        # See [Note: Criteria for allowing out-of-core tensor subclasses] for details on the criteria.
        self.tensor_subclasses_found: Dict[str, Type] = {}

    def load(self):
        """Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        """
        self.metastack = []
        self.stack: List[Any] = []
        self.append = self.stack.append
        read = self.read
        readline = self.readline
        dispatch = self.dispatch

        while True:
            key = read(1)
            if not key:
                raise EOFError
            assert isinstance(key, bytes_types)
            if key[0] in dispatch:
                if key[0] == STOP[0]:
                    return dispatch[key[0]](self)
                else:
                    dispatch[key[0]](self)
            else:
                raise RuntimeError(f"Unsupported operand {key[0]}")

    dispatch = {}

    # === Risky bytecode =======================================================

    def load_global(self):
        module = self.readline()[:-1].decode("utf-8")
        name = self.readline()[:-1].decode("utf-8")
        full_path = f"{module}.{name}"
        if full_path in _get_allowed_globals():
            self.append(_get_allowed_globals()[full_path])
        elif full_path in _get_user_allowed_globals():
            self.append(_get_user_allowed_globals()[full_path])
        else:
            # The logic in this branch handles user-defined tensor subclasses.
            # We can automatically allow and raise and error for anything that is not provably safe.
            # [Note: Criteria for allowing out-of-core tensor subclasses]
            # GLOBAL '<module>.<tensor subclass>' instructions will get the class and
            # push the string (not the actual type) while adding the type to the dictionary keyed
            # by the string onto the unpickler's stack if they satisfy the following conditions:
            # (1) The <module> that defines them is in `sys.modules`
            #     (we will use getattr_static to access it to ensure no code execution)
            # (2) They inherit from `torch.Tensor`
            # (2) The class is not overriding any of the `torch.Tensor` methods listed here:
            #     `__getattr__`, `__get__`, `__getattribute__`, `__setstate__`, `__set__`,
            #     and `tp_alloc`
            #     The methods that we ban overriding were selected in a test-driven manner
            #     by overriding every callable method on a tensor subclass and determinining
            #     which might get called during unpickling.
            # When executing REDUCE, the string will be appropriately converted back to the type only
            # for `torch._tensor._rebuild_from_type_v2` as other use of the class could use methods
            # we didn't audit.
            if module == "__builtin__":
                raise RuntimeError(
                    f"Unsupported global: GLOBAL {full_path} was not an allowed global by default. "
                    "Please use `torch.serialization.add_safe_globals` to allowlist this global "
                    "if you trust this class/function."
                )
            elif module not in modules:
                # TODO: add a link here to a doc that explains to users what we mean by trust
                raise RuntimeError(
                    f"Found GLOBAL `{full_path}` instruction in the pickle file but `{full_path}` was "
                    f"not in the pre-defined list of allowed globals that are considered safe by the "
                    "weights_only unpickler for rebuilding state_dicts. This is the expected behavior if "
                    f"`{full_path}` is a class or function that is not in the list of allowed globals "
                    f"If `{full_path}` is NOT a tensor subclass, you might consider"
                    "`torch.serialization.add_safe_globals` if it is appropriate. However, if it is a "
                    "user-defined tensor subclass not defined in the `torch` package, this error might arise "
                    f"as we expect `{module}` to be present in `sys.modules` (i.e. it "
                    "must be imported in the current environment), but this was not the case. "
                    f"If you intend to unpickle a tensor subclass `{full_path}` please import `{name}` from "
                    f"`{module}`. Note that having this imported will *only* allow the type `{full_path}` to "
                    "be passed as the second argument to `torch._tensor._rebuild_from_type_v2`, which should "
                    "enable the tensor subclass to be unpickled without any arbitrary code execution as long "
                    # If the user imports and these are overridden the next error will prompt them to use
                    # torch.serialization.add_safe_globals.
                    "a sa pre-defined list of methods called when unpickling are not overridden. In "
                    "particular, the methods are `__getattr__`, `__get__`, `__getattribute__`, `__setstate__`, "
                    "`__set__`, as well as the implementation of `tp_alloc`."
                )
            else:
                try:
                    class_type = getattr_static(modules[module], name)
                except AttributeError as e:
                    raise AttributeError(
                        "For safety during weights_only loading, we use inspect.getattr_state to "
                        f"get {name} from {module}, if {module} implements the descriptor protocol, "
                        "__getattr__ or __getattribute__ these will not be called."
                    ) from e
                # None of the objects here contain any data from the pickle so this is safe
                if isinstance(class_type, type) and issubclass(
                    class_type, torch.Tensor
                ):
                    # getattr is called by the getattr call in `_rebuild_from_type_v2`
                    custom_get_attribute = (
                        class_type.__getattribute__ is not torch.Tensor.__getattribute__
                    )
                    custom_get = getattr_static(class_type, "__get__", None) is not None
                    custom_get_attr = (
                        getattr_static(class_type, "__getattr__", None) is not None
                    )
                    # Tensor.__setstate__ might be called in `_rebuild_from_type_v2`
                    custom_set_state = (
                        class_type.__setstate__ is not torch.Tensor.__setstate__
                    )
                    # setattr is called in `torch._utils._set_obj_state`
                    custom_set_attr = class_type.__setattr__ is not object.__setattr__
                    custom_set = getattr_static(class_type, "__set__", None) is not None
                    # tp_alloc is called by `Tensor._rebuild_wrapper_subclass` and `Tensor.as_subclass`
                    has_custom_tp_alloc = not torch._C._check_tp_alloc_is_default(
                        class_type
                    )
                    custom_methods = {
                        "__getattribute__": custom_get_attribute,
                        "__getattr__": custom_get_attr,
                        "__get__": custom_get,
                        "__setattr__": custom_set_attr,
                        "__set__": custom_set,
                        "__setstate__": custom_set_state,
                        "tp_alloc": has_custom_tp_alloc,
                    }
                    if any(custom_methods.values()):
                        error = ""
                        for k, v in custom_methods.items():
                            error += f" {k}={v}"
                        raise RuntimeError(
                            f"Trying to unpickle tensor subclass `{full_path}` that has defined a custom "
                            f"version for one of these methods:{error}. Please check whether you trust these "
                            "methods and allowlist the subclass with `torch.serialization.add_safe_globals` if so."
                        )
                    # push the string full_path onto the stack (in REBUILD, there is special logic to
                    # access this from tensor_subclasses_found for rebuild_from_type_v2)
                    self.tensor_subclasses_found[full_path] = class_type
                    self.append(full_path)
                else:
                    raise RuntimeError(
                        f"Unsupported global: GLOBAL {full_path} was not an allowed global by default. "
                        "Please use `torch.serialization.add_safe_globals` to allowlist this global "
                        "if you trust this class/function."
                    )

    dispatch[GLOBAL[0]] = load_global

    def load_new_obj(self):
        args = self.stack.pop()
        cls = self.stack.pop()
        if cls is not torch.nn.Parameter:
            raise RuntimeError(f"Trying to instantiate unsupported class {cls}")
        self.append(torch.nn.Parameter(*args))

    dispatch[NEWOBJ[0]] = load_new_obj

    def load_reduce(self):
        args = self.stack.pop()
        func = self.stack[-1]
        if (
            func not in _get_allowed_globals().values()
            and func not in _get_user_allowed_globals().values()
        ):
            raise RuntimeError(
                f"Trying to call reduce for unrecognized function {func}"
            )
        # Special handling for tensor subclass type found in GLOBAL that is pushed
        # onto stack as str to prevent it from being used anywhere except the
        # second arg of _rebuild_from_type_v2 and within argument tuple for _rebuild_wrapper_subclass
        # _rebuild_from_type_v2 is called with args (func, type, func_args, state)
        # where both type and, when func is rebuild_wrapper_subclass, func_args[0] could be the subclass type
        # Since we pushed these subclass types onto the stack as strings, convert them to the actual
        # type here.
        if func is torch._tensor._rebuild_from_type_v2 and type(args[1]) is str:
            args_after = args[2:]
            if (
                args[0] is torch._utils._rebuild_wrapper_subclass
                and type(args[2][0]) is str
            ):
                new_arg_tuple = (self.tensor_subclasses_found[args[2][0]],) + args[2][
                    1:
                ]
                args_after = (new_arg_tuple,) + args[3:]
            args = args[:1] + (self.tensor_subclasses_found[args[1]],) + args_after
        self.stack[-1] = func(*args)

    dispatch[REDUCE[0]] = load_reduce

    def load_build(self):
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

    dispatch[BUILD[0]] = load_build

    # == Stack manipulation bytecode =======================================

    def load_append(self):
        item = self.stack.pop()
        list_obj = self.stack[-1]
        if type(list_obj) is not list:
            raise RuntimeError(f"Can only append to lists, but got {type(list_obj)}")
        list_obj.append(item)

    dispatch[APPEND[0]] = load_append

    def load_appends(self):
        items = self.pop_mark()
        list_obj = self.stack[-1]
        if type(list_obj) is not list:
            raise RuntimeError(f"Can only extend lists, but got {type(list_obj)}")
        list_obj.extend(items)

    dispatch[APPENDS[0]] = load_appends

    def load_setitem(self):
        (v, k) = (self.stack.pop(), self.stack.pop())
        self.stack[-1][k] = v

    dispatch[SETITEM[0]] = load_setitem

    def load_setitems(self):
        items = self.pop_mark()
        for i in range(0, len(items), 2):
            self.stack[-1][items[i]] = items[i + 1]

    dispatch[SETITEMS[0]] = load_setitems

    def load_mark(self):
        self.metastack.append(self.stack)
        self.stack = []
        self.append = self.stack.append

    dispatch[MARK[0]] = load_mark

    def load_tuple(self):
        items = self.pop_mark()
        self.append(tuple(items))

    dispatch[TUPLE[0]] = load_tuple

    def load_tuple1(self):
        self.stack[-1] = (self.stack[-1],)

    dispatch[TUPLE1[0]] = load_tuple1

    def load_tuple2(self):
        self.stack[-2:] = [(self.stack[-2], self.stack[-1])]

    dispatch[TUPLE2[0]] = load_tuple2

    def load_tuple3(self):
        self.stack[-3:] = [(self.stack[-3], self.stack[-2], self.stack[-1])]

    dispatch[TUPLE3[0]] = load_tuple3

    # == Basic type construction bytecode ==================================

    def load_none(self):
        self.append(None)

    dispatch[NONE[0]] = load_none

    def load_newfalse(self):
        self.append(False)

    dispatch[NEWFALSE[0]] = load_newfalse

    def load_newtrue(self):
        self.append(True)

    dispatch[NEWTRUE[0]] = load_newtrue

    def load_empty_tuple(self):
        self.append(())

    dispatch[EMPTY_TUPLE[0]] = load_empty_tuple

    def load_empty_list(self):
        self.append([])

    dispatch[EMPTY_LIST[0]] = load_empty_list

    def load_empty_dict(self):
        self.append({})

    dispatch[EMPTY_DICT[0]] = load_empty_dict

    def load_empty_set(self):
        self.append(set())

    dispatch[EMPTY_SET[0]] = load_empty_set

    def load_binint(self):
        self.append(unpack("<i", self.read(4))[0])

    dispatch[BININT[0]] = load_binint

    def load_binint1(self):
        self.append(self.read(1)[0])

    dispatch[BININT1[0]] = load_binint1

    def load_binint2(self):
        self.append(unpack("<H", self.read(2))[0])

    dispatch[BININT2[0]] = load_binint2

    def load_binfloat(self):
        self.append(unpack(">d", self.read(8))[0])

    dispatch[BINFLOAT[0]] = load_binfloat

    def load_binunicode(self):
        strlen = unpack("<I", self.read(4))[0]
        if strlen > maxsize:
            raise RuntimeError("String is too long")
        strval = str(self.read(strlen), "utf-8", "surrogatepass")
        self.append(strval)

    dispatch[BINUNICODE[0]] = load_binunicode

    def load_short_binstring(self):
        strlen = self.read(1)[0]
        strdata = self.read(strlen)
        if self.encoding != "bytes":
            strdata = strdata.decode(self.encoding, "strict")
        self.append(strdata)

    dispatch[SHORT_BINSTRING[0]] = load_short_binstring

    def load_binpersid(self):
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

    dispatch[BINPERSID[0]] = load_binpersid

    def load_binget(self):
        idx = self.read(1)[0]
        self.append(self.memo[idx])

    dispatch[BINGET[0]] = load_binget

    def load_long_binget(self):
        idx = unpack("<I", self.read(4))[0]
        self.append(self.memo[idx])

    dispatch[LONG_BINGET[0]] = load_long_binget

    def load_binput(self):
        i = self.read(1)[0]
        if i < 0:
            raise ValueError("negative argument")
        self.memo[i] = self.stack[-1]

    dispatch[BINPUT[0]] = load_binput

    def load_long_binput(self):
        i = unpack("<I", self.read(4))[0]
        if i < 0:
            raise ValueError("negative argument")
        self.memo[i] = self.stack[-1]

    dispatch[LONG_BINPUT[0]] = load_long_binput

    def load_long1(self):
        n = self.read(1)[0]
        data = self.read(n)
        self.append(decode_long(data))

    dispatch[LONG1[0]] = load_long1

    def load_proto(self):
        # Read and ignore proto version
        self.read(1)[0]

    dispatch[PROTO[0]] = load_proto

    def load_stop(self):
        rc = self.stack.pop()
        return rc

    dispatch[STOP[0]] = load_stop

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
