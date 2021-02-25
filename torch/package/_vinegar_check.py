import pickletools
from typing import List, Dict, Any, Callable, Optional


class PickleObjTree:
    def __init__(self, name, pos):
        self.pos = pos
        self.children = {}
        self.parent = None
        self.attr_name = None
        self.build_state = None
        self.name = name

    def __repr__(self):
        attr_name = self.attr_name if self.attr_name else "<no attr name>"
        return f"({self.name}, {self.pos}, {attr_name})"


class TrackedObj(PickleObjTree):
    def __init__(self, name, pos):
        super().__init__(name, pos)


class TrackedTuple(PickleObjTree):
    def __init__(self, tup, pos):
        super().__init__("<tuple>", pos)
        for idx, el in enumerate(tup):
            self.children[idx] = el


class TrackedDict(PickleObjTree):
    def __init__(self, pos):
        super().__init__("<dict>", pos)

    def __setitem__(self, key, value):
        self.children[key] = value

    def __getitem__(self, key):
        return self.children[key]

    def __iter__(self):
        return self.children.__iter__()

    def keys(self):
        return self.children.keys()

    def values(self):
        return self.children.values()

    def items(self):
        return self.children.items()


class TrackedList(PickleObjTree):
    def __init__(self, pos):
        super().__init__("<list>", pos)

    def append(self, el):
        self.children[len(self.children)] = el

    def extend(self, it):
        for i in it:
            self.children[len(self.children)] = i

    def __len__(self):
        return len(self.children)


class PickleAnalyzer:
    def __init__(self, pickle_bytes: bytes):
        self.pickle_bytes = pickle_bytes

    def scan_binary(self, binary: bytes = None) -> PickleObjTree:
        if not binary:
            binary = self.pickle_bytes
        i = pickletools.genops(binary)

        self.memo: Dict[int, Any] = {}
        self.metastack: List[Any] = []
        self.stack: List[Any] = []
        self.append = self.stack.append
        self.used = set()
        for opcode, arg, pos in i:
            name = opcode.name
            self.used.add(name)
            self.dispatch[name](self, arg, pos)

        ret = self.stack.pop()
        ret.attr_name = "<pickled root object>"
        assert len(self.stack) == 0
        return ret

    def pop_mark(self):
        items = self.stack
        self.stack = self.metastack.pop()
        self.append = self.stack.append
        return items

    def load_simple_push(self, arg, pos):
        self.append(arg)

    def load_empty_tuple(self, arg, pos):
        self.append(TrackedTuple((), pos))

    def load_empty_dict(self, arg, pos):
        self.append(TrackedDict(pos))

    def load_empty_list(self, arg, pos):
        self.append(TrackedList(pos))

    def load_tuple(self, arg, pos):
        items = self.pop_mark()
        self.append(TrackedTuple(items, pos))

    def load_tuple1(self, arg, pos):
        self.stack[-1] = TrackedTuple(self.stack[-1:], pos)

    def load_tuple2(self, arg, pos):
        self.stack[-2:] = [TrackedTuple(self.stack[-2:], pos)]

    def load_tuple3(self, arg, pos):
        self.stack[-3:] = [TrackedTuple(self.stack[-3:], pos)]

    def load_none(self, arg, pos):
        self.append(None)

    def load_newtrue(self, arg, pos):
        self.append(True)

    def load_newfalse(self, arg, pos):
        self.append(False)

    def load_binpersid(self, arg, pos):
        pass

    def load_memo_put(self, arg, pos):
        self.memo[arg] = self.stack[-1]

    def load_memo_get(self, arg, pos):
        self.append(self.memo[arg])

    def load_global(self, arg, pos):
        # GLOBAL is returned a string, not class/fn
        self.append(arg)

    def load_reduce(self, arg, pos):
        args = self.stack.pop()
        func = self.stack.pop()
        if func == "collections OrderedDict":
            # hack to make this work
            self.append(TrackedDict(pos))
        else:
            ret = TrackedObj(func, pos)
            children = {}
            for i, arg in enumerate(args.children.values()):
                arg_name = f"reduce_arg_{i}"
                children[arg_name] = arg
                if isinstance(arg, PickleObjTree):
                    arg.attr_name = arg_name
                    arg.parent = ret
            ret.children = children
            self.append(ret)

    def load_newobj(self, arg, pos):
        args = self.stack.pop()
        klass = self.stack.pop()
        obj = TrackedObj(klass, pos)
        self.append(obj)

    def load_build(self, arg, pos):
        state = self.stack.pop()
        obj = self.stack[-1]
        if isinstance(state, TrackedDict):
            # this is a regular __dict__ update.
            assert len(obj.children) == 0
            obj.children.update(state)

            # Annotate any objs with parent and attr information for better unwinding.
            for k, v in state.items():
                if not isinstance(v, PickleObjTree):
                    continue
                v.parent = obj
                v.attr_name = k
        elif isinstance(state, TrackedTuple):
            # dealing with object that has __slots__
            assert len(state.children) == 2
            assert type(state.children[1]) == TrackedDict
            for k, v in state.children[1].items():
                if not isinstance(v, PickleObjTree):
                    continue
                obj.children[k] = v
                v.parent = obj
                v.attr_name = k
        else:
            # This is something else, either a slotstate builder or setstate call.
            # We don't super care about this case
            obj.build_state = state

    def load_setitem(self, arg, pos):
        value = self.stack.pop()
        key = self.stack.pop()
        d = self.stack[-1]
        if isinstance(value, PickleObjTree):
            value.parent = d
            value.attr_name = key
        d[key] = value

    def load_mark(self, arg, pos):
        self.metastack.append(self.stack)
        self.stack = []
        self.append = self.stack.append

    def load_proto(self, arg, pos):
        assert arg == 3

    def load_stop(self, arg, pos):
        pass

    def load_appends(self, arg, pos):
        items = self.pop_mark()
        list_ = self.stack[-1]
        idx = len(list_)
        for value in items:
            if isinstance(value, PickleObjTree):
                value.parent = list_
                value.attr_name = f"<list element {idx}>"
                idx += 1

        list_.extend(items)

    def load_append(self, arg, pos):
        value = self.stack.pop()
        list_ = self.stack[-1]
        if isinstance(value, PickleObjTree):
            value.parent = list_
            value.attr_name = f"<list element {len(list_)}>"
        list_.append(value)

    def load_setitems(self, arg, pos):
        items = self.pop_mark()
        dict = self.stack[-1]
        for i in range(0, len(items), 2):
            k = items[i]
            v = items[i + 1]
            dict[k] = v
            if isinstance(v, PickleObjTree):
                v.parent = dict
                v.attr_name = k

    dispatch = {}
    dispatch["BINUNICODE"] = load_simple_push
    dispatch["BINFLOAT"] = load_simple_push
    dispatch["BININT"] = load_simple_push
    dispatch["BININT1"] = load_simple_push
    dispatch["BININT2"] = load_simple_push
    dispatch["BINBYTES"] = load_simple_push
    dispatch["SHORT_BINBYTES"] = load_simple_push
    dispatch["EMPTY_TUPLE"] = load_empty_tuple
    dispatch["EMPTY_DICT"] = load_empty_dict
    dispatch["EMPTY_LIST"] = load_empty_list
    dispatch["TUPLE"] = load_tuple
    dispatch["TUPLE1"] = load_tuple1
    dispatch["TUPLE2"] = load_tuple2
    dispatch["TUPLE3"] = load_tuple3
    dispatch["NEWTRUE"] = load_newtrue
    dispatch["NEWFALSE"] = load_newfalse
    dispatch["NONE"] = load_none
    dispatch["BINPERSID"] = load_binpersid
    dispatch["BINPUT"] = load_memo_put
    dispatch["LONG_BINPUT"] = load_memo_put
    dispatch["BINGET"] = load_memo_get
    dispatch["LONG_BINGET"] = load_memo_get
    dispatch["GLOBAL"] = load_global
    dispatch["REDUCE"] = load_reduce
    dispatch["NEWOBJ"] = load_newobj
    dispatch["BUILD"] = load_build
    dispatch["SETITEM"] = load_setitem
    dispatch["MARK"] = load_mark
    dispatch["PROTO"] = load_proto
    dispatch["STOP"] = load_stop
    dispatch["APPENDS"] = load_appends
    dispatch["APPEND"] = load_append
    dispatch["SETITEMS"] = load_setitems


def scan(obj: PickleObjTree, pred: Callable[[str], bool], ret: List[PickleObjTree]):
    if isinstance(obj, PickleObjTree):
        if isinstance(obj, TrackedObj) and pred(obj.name):
            if obj not in ret:
                ret += [obj]
        for child in obj.children.values():
            scan(child, pred, ret)


def find_pickle_dependencies(pickle: PickleObjTree, dependend_modules: List[str]):
    matches: List[PickleObjTree] = []
    dependend_modules.sort(key=len)
    for module_name in dependend_modules:
        scan(pickle, lambda x: module_name in x, matches)
    return matches


def format_stack_mocked(obj_tree: PickleObjTree):
    obj_path = []
    cur: Optional[PickleObjTree] = obj_tree
    while cur is not None:
        obj_path.append(cur)
        cur = cur.parent
    obj_path.reverse()

    def strip_modules_from_path(obj_path):
        return [el for el in obj_path if el.attr_name != "_modules"]

    obj_path = strip_modules_from_path(obj_path)

    result = "MOCKED OBJECT FOUND AT: "
    result += ".".join(
        [obj.attr_name if obj.attr_name is not None else "<noinfo>" for obj in obj_path]
    )
    result += "\nObjects on path:\n"
    for obj in obj_path:
        result += f"    {obj.attr_name}\n"
        result += f"      type: '{obj.name}'\n"
    module_name, _sep, class_name = obj_tree.name.partition(" ")
    result += f"\nType '{class_name}' was flagged because the module '{module_name}' was mocked during export.\n"
    return result


def scan_pickle_for_dependencies(
    pickle_bytes: bytes, mocked_modules: List[str]
) -> List[PickleObjTree]:
    pickled_obj = PickleAnalyzer(pickle_bytes).scan_binary()
    mocked_objects = find_pickle_dependencies(pickled_obj, mocked_modules)
    return mocked_objects


def mocked_objects_str(mocked_objects: List[PickleObjTree], print_limit: int = 1):
    output = ""
    if len(mocked_objects) > 0:
        output += f"WARNING: found {len(mocked_objects)} mocked object(s) referenced in pickle. "
        output += f"Printing first {print_limit} objects.\n"
        for i, mocked_obj in enumerate(mocked_objects):
            if i < print_limit:
                output += format_stack_mocked(mocked_obj)
    elif print_limit > 0:
        output = "Found zero mocked objects referenced in pickle."
    return output
