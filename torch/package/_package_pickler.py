"""isort:skip_file"""
from struct import pack
from types import FunctionType
from pickle import _compat_pickle, _tuplesize2code, _extension_registry, _getattribute, _Pickler  # type: ignore[attr-defined]
from itertools import islice
import io
from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer

from pickle import (
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    STACK_GLOBAL,
    Pickler,
    PicklingError,
    EMPTY_LIST,
    EMPTY_TUPLE,
    TUPLE,
    MARK,
    LIST,
    APPEND,
    EMPTY_DICT,
    DICT,
    SETITEM,
    POP,
    POP_MARK,
    EMPTY_SET,
)

class PackagePickler(_Pickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    dispatch = _Pickler.dispatch.copy()

    def __init__(self, importer: Importer, *args, **kwargs):
        self.importer = importer
        super().__init__(*args, **kwargs)

    def save_global(self, obj, name=None):
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write
        memo = self.memo

        # CHANGED: import module from module environment instead of __import__
        try:
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from None

        module = self.importer.import_module(module_name)
        _, parent = _getattribute(module, name)
        # END CHANGED

        if self.proto >= 2:
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition(".")[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:
            self.save(module_name)
            self.save(name)
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))
        elif self.proto >= 3:
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            if self.fix_imports:
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError:
                raise PicklingError(
                    "can't pickle global identifier '%s.%s' using "
                    "pickle protocol %i" % (module, name, self.proto)
                ) from None

        self.memoize(obj)

    dispatch[FunctionType] = save_global

class DebugInfo:
    def format(self):
        raise AssertionError(f"unknown debug type {self}")
    def __repr__(self):
        return f"DebugInfo({self.__class__.__name__})"

class RootObject(DebugInfo):
    def __init__(self, obj):
        self.obj = obj

    def format(self):
        return f"<pickled object> ({type(self.obj)})"

class ObjectAttribute(DebugInfo):
    def __init__(self, attr_name, attr_value):
        self.attr_name = attr_name
        self.attr_value = attr_value

    def format(self):
        return f"  .{self.attr_name} ({type(self.attr_value)})"

class ListElement(DebugInfo):
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

    def format(self):
        return f"  <object @ idx {self.idx}> ({type(self.value)})"

class TupleElement(DebugInfo):
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

    def format(self):
        return f"  <object @ idx {self.idx}> ({type(self.value)})"

class DictElement(DebugInfo):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def format(self):
        return f"  <object @ key {self.key}> ({type(self.value)})"

class SetElement(DebugInfo):
    def __init__(self, obj):
        self.obj = obj

    def format(self):
        return f"  <object {self.obj}> ({type(self.obj)})"

class SetElementPreProtocol4Flag(DebugInfo):
    # for protocol < 4 converts sets into a tupled list, so we flag and error is handled in trace
    def __init__(self, proto):
        pass

class TypeElement(DebugInfo):
    def __init__(self, obj):
        self.obj = obj

    def format(self):
        return f"  type {type(self.obj)} (from object {(self.obj)})"

class DebugPickler(PackagePickler):
    """A pickler that implements debug tracing functionality to better
    """
    def __init__(self, *args, **kwargs):
        self.obj_stack = []
        super().__init__(*args, **kwargs)

    dispatch = _Pickler.dispatch.copy()

    def save_list(self, obj):
        if self.bin:
            self.write(EMPTY_LIST)
        else:  # proto 0 -- can't use EMPTY_LIST
            self.write(MARK + LIST)

        self.memoize(obj)
        save = self.save
        write = self.write

        for idx, x in enumerate(obj):
            self.obj_stack.append(ListElement(idx, x))
            save(x)
            write(APPEND)
            self.obj_stack.pop()

    dispatch[list] = save_list

    def save_dict(self, obj):
        if self.bin:
            self.write(EMPTY_DICT)
        else:  # proto 0 -- can't use EMPTY_DICT
            self.write(MARK + DICT)

        self.memoize(obj)
        save = self.save
        write = self.write

        for k, v in obj.items():
            self.obj_stack.append(DictElement(k, v))
            save(k)
            save(v)
            write(SETITEM)
            self.obj_stack.pop()

    dispatch[dict] = save_dict

    def save_tuple(self, obj):
        if not obj:  # tuple is empty
            if self.bin:
                self.write(EMPTY_TUPLE)
            else:
                self.write(MARK + TUPLE)
            return

        n = len(obj)
        save = self.save
        memo = self.memo
        idx = 0
        if n <= 3 and self.proto >= 2:
            for element in obj:
                self.obj_stack.append(TupleElement(idx, element))
                save(element)
                self.obj_stack.pop()
                idx += 1
            # Subtle.  Same as in the big comment below.
            if id(obj) in memo:
                get = self.get(memo[id(obj)][0])
                self.write(POP * n + get)
            else:
                self.write(_tuplesize2code[n])
                self.memoize(obj)
            return

        # proto 0 or proto 1 and tuple isn't empty, or proto > 1 and tuple
        # has more than 3 elements.
        write = self.write
        write(MARK)
        for element in obj:
            self.obj_stack.append(TupleElement(idx, element))
            save(element)
            self.obj_stack.pop()
            idx += 1

        if id(obj) in memo:
            # Subtle.  d was not in memo when we entered save_tuple(), so
            # the process of saving the tuple's elements must have saved
            # the tuple itself:  the tuple is recursive.  The proper action
            # now is to throw away everything we put on the stack, and
            # simply GET the tuple (it's already constructed).  This check
            # could have been done in the "for element" loop instead, but
            # recursive tuples are a rare thing.
            get = self.get(memo[id(obj)][0])
            if self.bin:
                write(POP_MARK + get)
            else:   # proto 0 -- POP_MARK not available
                write(POP * (n+1) + get)
            return

        # No recursion.
        write(TUPLE)
        self.memoize(obj)

    dispatch[tuple] = save_tuple

    def save_set(self, obj, set_type=set):
        save = self.save
        write = self.write

        if self.proto < 4:
            self.obj_stack.append(SetElementPreProtocol4Flag(self.proto))
            self.save_reduce(set_type, (list(obj),), obj=obj)
            self.obj_stack.pop()
            return

        write(EMPTY_SET)
        self.memoize(obj)

        it = iter(obj)
        while True:
            batch = list(islice(it, self._BATCHSIZE))
            n = len(batch)
            if n > 0:
                write(MARK)
                for item in batch:
                    self.obj_stack.append(SetElement(item))
                    save(item)
                    self.obj_stack.pop()
                write(ADDITEMS)
            if n < self._BATCHSIZE:
                return
    dispatch[set] = save_set

    def save_frozenset(self, obj, settype=set):
        self.save_set(obj, set_type = frozenset)
    dispatch[frozenset] = save_frozenset

    def save(self, obj, save_persistent_id=True):
        self.obj_stack.append(obj)
        super().save(obj, save_persistent_id)
        self.obj_stack.pop()



    def format_trace(self):
        stack = self.obj_stack
        traced_stack = []
        idx_to_skip = set()
        for idx, obj in enumerate(stack):
            if idx in idx_to_skip:
                continue

            if isinstance(obj, dict) and idx > 0:
                # Fold:
                # [
                #   obj,
                #   obj's __dict__,
                #   DictElement(k, v)
                #   obj for v
                # ]
                # into:
                # [
                #   obj,
                #   (obj attribute, k, v)
                # ]
                prev_obj = stack[idx - 1]
                if getattr(prev_obj, "__dict__", None) is obj:
                    dict_element = stack[idx + 1]
                    traced_stack.append(ObjectAttribute(dict_element.key, dict_element.value))
                    idx_to_skip.add(idx + 1)
                    idx_to_skip.add(idx + 2)
                    continue
            if isinstance(obj, (ListElement, DictElement, TupleElement, SetElement)):
                # Fold [ListElement, obj] into just [ListElement]
                traced_stack.append(obj)
                idx_to_skip.add(idx + 1)
                continue
            elif isinstance(obj, SetElementPreProtocol4Flag):
                # handle sets for protocol < 4 where set is converted to ([set],) / a list nested in a tuple
                assert isinstance(stack[idx + 2], TupleElement)
                assert isinstance(stack[idx + 4], ListElement)
                idx_to_skip.add(idx + 1)
                idx_to_skip.add(idx + 2)
                idx_to_skip.add(idx + 3)
                idx_to_skip.add(idx + 4)
                idx_to_skip.add(idx + 5)
                bad_element = stack[idx + 4].value
                traced_stack.append(SetElement(bad_element))
                continue
            traced_stack.append(obj)

        stack = traced_stack
        stack[0] = RootObject(stack[0])

        msg = io.StringIO()
        for obj in stack:
            if not isinstance(obj, DebugInfo):
                msg.write(f"  {repr(obj)}\n")
                continue
            msg.write(obj.format())
            msg.write("\n")
        return msg.getvalue()


def debug_dumps(importer, obj, protocol=4):
    f = io.BytesIO()
    p = DebugPickler(importer, f, protocol=protocol)
    try:
        p.dump(obj)
    except (TypeError, PicklingError) as e:
        # print(f"{str(e)}.\n\n"
        #     "We think the problematic object is found at:\n"
        #     f"{p.format_trace()}")
        raise PicklingError(
            f"{str(e)}.\n\n"
            "We think the problematic object is found at:\n"
            f"{p.format_trace()}"
        ) from e
    else:
        raise ValueError(f"We expected pickling of {obj} to throw an exception")

def create_pickler(data_buf, importer, protocol=4):
    if importer is sys_importer:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return Pickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)
