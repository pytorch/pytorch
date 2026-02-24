# Design: TypeVariableTracker — Mirroring CPython's PyObject/PyTypeObject in Dynamo

**Status**: Proposal / Discussion
**Author**: anijain (with Claude)
**Date**: 2026-02-23

## Problem Statement

Dynamo's VariableTracker (VT) hierarchy fuses instance data and type behavior
into a single class hierarchy. Each VT subclass (`ListVariable`,
`TensorVariable`, `UserDefinedObjectVariable`, etc.) re-implements behavioral
methods (`var_getattr`, `call_method`, `call_function`, etc.) independently.
This causes:

1. **CPython behavioral gaps**: Each VT subclass has its own approximation of
   CPython semantics (attribute lookup, binary ops, descriptor protocol, etc.).
   Fixing a gap in one VT doesn't fix it in others.

2. **Code duplication**: The same CPython algorithm (e.g., the 3-step attribute
   lookup in `PyObject_GenericGetAttr`) is reimplemented across multiple VT
   classes with overlapping but divergent logic.

3. **Subclass cliff**: When a user subclasses a built-in type (e.g.,
   `class MyList(list)`), the instance falls into `UserDefinedObjectVariable`
   and loses all optimized handling that `ListVariable` provides.

4. **Large isinstance cascades**: `side_effects.py` codegen and
   `BuiltinVariable` dispatch use `isinstance` chains to determine behavior,
   because VTs don't carry structured type metadata.

## CPython Background

### PyObject — the universal instance header

Every CPython object starts with:

```c
typedef struct _object {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;    // pointer to the type
} PyObject;
```

A concrete type like `PyListObject` extends this with its data:

```c
typedef struct {
    PyObject ob_base;         // refcnt + type pointer
    PyObject **ob_item;       // the array of items
    Py_ssize_t allocated;
} PyListObject;
```

The key insight: `ob_type` at a fixed offset enables polymorphic dispatch.
You pass `PyObject*` everywhere and dispatch through `ob_type`.

### PyTypeObject — the type descriptor / vtable

`PyTypeObject` is itself a `PyObject` (types are objects). It holds:

- **Identity**: `tp_name`, `tp_basicsize`, `tp_itemsize`
- **Inheritance**: `tp_base`, `tp_mro`, `tp_bases`
- **Slot functions** (the vtable):
  - Core: `tp_getattro`, `tp_setattro`, `tp_call`, `tp_hash`, `tp_repr`,
    `tp_str`, `tp_richcompare`, `tp_iter`, `tp_iternext`, `tp_init`, `tp_new`
  - Number protocol: `tp_as_number->nb_add`, `nb_multiply`, `nb_bool`, etc.
  - Sequence protocol: `tp_as_sequence->sq_length`, `sq_item`, `sq_contains`,
    `sq_concat`, `sq_repeat`, etc.
  - Mapping protocol: `tp_as_mapping->mp_length`, `mp_subscript`,
    `mp_ass_subscript`
  - Descriptor protocol: `tp_descr_get` (`__get__`), `tp_descr_set` (`__set__`)
- **Type dict**: `tp_dict` — class-level attributes (methods, class variables)
- **Flags**: `tp_flags` — `Py_TPFLAGS_HEAPTYPE`, `Py_TPFLAGS_IMMUTABLETYPE`,
  `Py_TPFLAGS_BASETYPE`, etc.
- **Dict offset**: `tp_dictoffset` — whether/where instances have `__dict__`

### Attribute lookup: `PyObject_GenericGetAttr` (default `tp_getattro`)

```
1. type_attr = walk type(obj).__mro__, looking in each type's __dict__
2. if type_attr is a DATA descriptor (has __get__ AND __set__):
       return type_attr.__get__(obj, type(obj))
3. if name in obj.__dict__:
       return obj.__dict__[name]
4. if type_attr is a NON-DATA descriptor (has __get__ but NOT __set__):
       return type_attr.__get__(obj, type(obj))
5. if type_attr exists (plain class variable):
       return type_attr
6. raise AttributeError → __getattr__ fallback
```

Priority: **data descriptors > instance dict > non-data descriptors > plain class vars**.

### Attribute setting: `PyObject_GenericSetAttr` (default `tp_setattro`)

Simpler than getattr:

```
1. type_attr = walk type(obj).__mro__
2. if type_attr is a DATA descriptor (has __set__):
       type_attr.__set__(obj, value)
       return
3. if obj has __dict__:
       obj.__dict__[name] = value
       return
4. raise AttributeError
```

Non-data descriptors are invisible to setattr. This is why `obj.x = 5` can
shadow a method — methods (functions) are non-data descriptors.

### Descriptor protocol

The descriptor protocol is what connects functions to bound methods, and what
makes `property`, `classmethod`, `staticmethod`, and `__slots__` work:

- **Data descriptor**: type has both `tp_descr_get` and `tp_descr_set`
  (e.g., `property`, `member_descriptor` from `__slots__`)
- **Non-data descriptor**: type has `tp_descr_get` but NOT `tp_descr_set`
  (e.g., `function`, `staticmethod`, `classmethod`)

When `generic_getattr` finds a function in the MRO, it calls
`function.__get__(obj, type(obj))` which returns a bound method. This is NOT
special-cased — it's the descriptor protocol applied uniformly.

### Functions and Methods in CPython

A `PyFunctionObject` has `ob_type = &PyFunction_Type`. `PyFunction_Type` has:
- `tp_call` — invokes the function (sets up frame, runs bytecode)
- `tp_descr_get` — creates bound methods: `function.__get__(obj, cls)` returns
  a `PyMethodObject`
- `tp_descr_set = NULL` — function is a NON-data descriptor

A `PyMethodObject` has `ob_type = &PyMethod_Type`. It stores:
- `im_func` (`__func__`) — the underlying function
- `im_self` (`__self__`) — the bound instance

`PyMethod_Type` has `tp_call` that does:
`call im_func with (im_self, *args, **kwargs)`.

**Critical**: `PyMethodObject` does NOT inherit from `PyFunctionObject`. They
are separate types connected by the descriptor protocol.

### Slot inheritance via `PyType_Ready`

When a type is initialized:
1. If `tp_base` is NULL, default to `object`
2. For each NULL slot, copy from `tp_base` (C-level slot inheritance)
3. Compute MRO via C3 linearization
4. Build `tp_dict` from `tp_methods`, `tp_members`, `tp_getset`

### Heap types vs static types

- **Static types** (`PyList_Type`, `PyLong_Type`): statically allocated C
  structs, `tp_flags` does NOT have `Py_TPFLAGS_HEAPTYPE`, immutable `tp_dict`
- **Heap types** (from `class Foo: ...`): dynamically allocated, mutable
  `tp_dict`, reference-counted, have `Py_TPFLAGS_HEAPTYPE`

## Current Dynamo State

### VT hierarchy structure

```
VariableTracker (base)
├── source: Source | None
├── mutation_type: MutationType | None
├── python_type() → type              # returns real Python type, not a VT
├── var_getattr(tx, name) → VT        # each subclass overrides
├── call_method(tx, name, args, kw)   # each subclass overrides
├── call_function(tx, args, kw)       # each subclass overrides
│
├── UserDefinedObjectVariable
│   ├── value: object                 # the real Python object
│   ├── value_type: type              # type(value)
│   ├── var_getattr: ~400 lines implementing PyObject_GenericGetAttr
│   └── (descriptor handling, __getattr__ fallback, etc.)
│
├── UserDefinedClassVariable
│   ├── value: type                   # the real class
│   ├── var_getattr: ~300 lines implementing type_getattro
│   └── (metaclass descriptors, MRO walk, etc.)
│
├── ListVariable (via BaseListVariable → CommonListMethodsVariable)
│   ├── items: list[VT]              # the payload
│   ├── call_method: ~200 lines of method dispatch
│   ├── var_getattr: handles __class__
│   └── python_type() → list
│
├── TensorVariable
│   ├── proxy: fx.Proxy
│   ├── var_getattr: tensor-specific
│   └── call_method: tensor-specific
│
├── UserFunctionVariable (via BaseUserFunctionVariable)
│   ├── fn: types.FunctionType        # the real function
│   ├── call_function: inlines the function
│   ├── var_getattr: fn_var_getattr (inspect.getattr_static)
│   └── python_type() → types.FunctionType
│
└── UserMethodVariable(UserFunctionVariable)  # ← WRONG: method inherits from function
    ├── obj: VariableTracker           # __self__
    ├── source_fn: Source | None
    ├── call_function: prepends self, delegates
    └── var_getattr: hardcodes __self__, __func__
```

### Where behavior lives (the problem)

| Operation | CPython | Dynamo today |
|---|---|---|
| Attribute lookup | `ob_type->tp_getattro` (on type) | `var_getattr` override (on each VT subclass) |
| Method call | `tp_call` on bound method type | `call_method` override (on each VT subclass) |
| `obj()` | `ob_type->tp_call` (on type) | `call_function` override (on each VT subclass) |
| `obj + other` | `ob_type->tp_as_number->nb_add` | `BuiltinVariable` isinstance cascade |
| `len(obj)` | `ob_type->tp_as_sequence->sq_length` | `BuiltinVariable` isinstance cascade |
| `iter(obj)` | `ob_type->tp_iter` | `call_method("__iter__")` per VT |
| `hash(obj)` | `ob_type->tp_hash` | per-VT `is_python_hashable` / `get_python_hash` |
| Descriptor `__get__` | `ob_type->tp_descr_get` on the descriptor's type | hardcoded in `_resolve_type_attr` |
| Reconstruction | N/A | isinstance cascade in `side_effects.py` |

### Specific pain points

1. **`UserMethodVariable` inherits from `UserFunctionVariable`**: In CPython,
   methods and functions are separate types. The inheritance exists only to
   share `fn` field and call logic, but conflates type identities.

2. **`python_type()` returns a real type, not a VT**: There's no way to
   represent a traced/guarded type. `ob_type` is always a concrete Python
   value, not a tracked variable.

3. **No unified `TypeVariableTracker`**: `UserDefinedClassVariable` handles
   user classes, but built-in types like `list`, `int`, `torch.Tensor` are
   represented as `ConstantVariable(list)` or handled by bespoke VT classes.

4. **Attribute lookup is duplicated**: `UserDefinedObjectVariable.var_getattr`,
   `TensorVariable.var_getattr`, `ListVariable.var_getattr`, etc. each
   reimplement overlapping subsets of `PyObject_GenericGetAttr`.

5. **Instance dict not first-class**: `UserDefinedObjectVariable` accesses
   `self.value.__dict__` (the real dict) at trace time. The instance dict
   isn't a tracked VT field.

6. **Reconstruction uses isinstance cascades**: `side_effects.codegen_update_mutated`
   is a ~350-line isinstance chain because it must reverse-engineer type
   information from VT class identity.

## Proposed Design

### Core principle

**Separate instance data from type behavior.** Behavior lives on
`TypeVariableTracker` (the `PyTypeObject` equivalent). VT subclasses carry
only instance data. The base `VariableTracker` delegates to `ob_type_vt`.

### `TypeVariableTracker`

```python
@dataclasses.dataclass
class TypeFlags:
    has_instance_dict: bool = False    # tp_dictoffset != 0
    is_heaptype: bool = False          # Py_TPFLAGS_HEAPTYPE
    is_immutable_type: bool = False    # Py_TPFLAGS_IMMUTABLETYPE
    is_sequence: bool = False          # tp_as_sequence != NULL
    is_mapping: bool = False           # tp_as_mapping != NULL
    is_hashable: bool = True           # tp_hash != NULL


# Slot function signature: (tx, instance_vt, *args, **kwargs) -> VariableTracker
Slot = Callable[..., VariableTracker]


class TypeVariableTracker(VariableTracker):
    """
    Mirrors PyTypeObject. Behavior lives here, not on instance VTs.
    Built-in types get pre-built singletons. User-defined (heap) types
    are constructed dynamically with guards on tp_dict and MRO.
    """
    python_cls: type
    tp_base: TypeVariableTracker | None
    tp_mro: tuple[TypeVariableTracker, ...]
    tp_flags: TypeFlags
    tp_dict: dict[str, VariableTracker]

    # ---- Slot table ----
    # Core
    tp_getattro: Slot | None = None
    tp_setattro: Slot | None = None
    tp_call: Slot | None = None
    tp_init: Slot | None = None
    tp_new: Slot | None = None
    tp_repr: Slot | None = None
    tp_str: Slot | None = None
    tp_hash: Slot | None = None
    tp_bool: Slot | None = None        # nb_bool in CPython
    tp_iter: Slot | None = None
    tp_iternext: Slot | None = None
    tp_richcompare: Slot | None = None

    # Sequence protocol
    sq_length: Slot | None = None
    sq_item: Slot | None = None
    sq_ass_item: Slot | None = None
    sq_contains: Slot | None = None
    sq_concat: Slot | None = None
    sq_repeat: Slot | None = None
    sq_inplace_concat: Slot | None = None
    sq_inplace_repeat: Slot | None = None

    # Mapping protocol
    mp_length: Slot | None = None
    mp_subscript: Slot | None = None
    mp_ass_subscript: Slot | None = None

    # Number protocol (expand as needed)
    nb_add: Slot | None = None
    nb_radd: Slot | None = None
    nb_multiply: Slot | None = None
    nb_negative: Slot | None = None
    # ... etc

    # Descriptor protocol (for when instances of this type ARE descriptors)
    tp_descr_get: Slot | None = None
    tp_descr_set: Slot | None = None

    # Codegen hooks (Dynamo-specific, no CPython equivalent)
    tp_codegen_sync: Callable | None = None    # how to replay value mutations
    tp_codegen_new: Callable | None = None     # how to emit __new__ bytecode

    def lookup_mro(self, name: str) -> VariableTracker | None:
        for type_vt in self.tp_mro:
            if name in type_vt.tp_dict:
                return type_vt.tp_dict[name]
        return None

    def is_data_descriptor(self, attr_vt: VariableTracker) -> bool:
        attr_type_vt = attr_vt.ob_type_vt
        return (attr_type_vt.tp_descr_get is not None
                and attr_type_vt.tp_descr_set is not None)

    def is_subtype_of(self, other: TypeVariableTracker) -> bool:
        return other in self.tp_mro
```

### `VariableTracker` base — gains `ob_type_vt` and delegates

```python
class VariableTracker:
    ob_type_vt: TypeVariableTracker
    source: Source | None
    mutation_type: MutationType | None

    def python_type(self):
        return self.ob_type_vt.python_cls

    def var_getattr(self, tx, name):
        return self.ob_type_vt.tp_getattro(tx, self, name)

    def call_function(self, tx, args, kwargs):
        if self.ob_type_vt.tp_call is None:
            raise TypeError(...)
        return self.ob_type_vt.tp_call(tx, self, args, kwargs)

    def call_method(self, tx, name, args, kwargs):
        # Check slots first (direct C-level dispatch)
        slot = self.ob_type_vt.resolve_slot_for_method(name)
        if slot is not None:
            return slot(tx, self, *args, **kwargs)
        # Fall back to attribute lookup
        method_vt = self.var_getattr(tx, name)
        return method_vt.call_function(tx, args, kwargs)
```

Where `resolve_slot_for_method` maps dunder names to C-level slots:

```python
_METHOD_TO_SLOT = {
    "__getitem__": "mp_subscript",  # CPython prefers mapping over sequence
    "__setitem__": "mp_ass_subscript",
    "__len__": "sq_length",
    "__contains__": "sq_contains",
    "__add__": "sq_concat",         # for sequences; nb_add for numbers
    "__mul__": "sq_repeat",         # for sequences; nb_multiply for numbers
    "__iadd__": "sq_inplace_concat",
    "__iter__": "tp_iter",
    "__next__": "tp_iternext",
    "__call__": "tp_call",
    "__hash__": "tp_hash",
    "__bool__": "tp_bool",
    "__repr__": "tp_repr",
    # ...
}
```

### `generic_getattr` — shared default for `tp_getattro`

```python
def generic_getattr(tx, instance_vt, name):
    """
    PyObject_GenericGetAttr. One implementation replaces var_getattr
    overrides on ListVariable, UserDefinedObjectVariable, etc.
    """
    type_vt = instance_vt.ob_type_vt

    # Step 1: Walk MRO
    type_attr = type_vt.lookup_mro(name)

    # Step 2: Data descriptor priority
    if type_attr is not None and type_vt.is_data_descriptor(type_attr):
        return type_attr.ob_type_vt.tp_descr_get(tx, type_attr, instance_vt, type_vt)

    # Step 3: Instance dict
    if type_vt.tp_flags.has_instance_dict:
        instance_dict = instance_vt.get_instance_dict()
        if instance_dict is not None and name in instance_dict:
            return instance_dict[name]

    # Step 4-5: Non-data descriptor or plain class attr
    if type_attr is not None:
        if type_attr.ob_type_vt.tp_descr_get is not None:
            return type_attr.ob_type_vt.tp_descr_get(
                tx, type_attr, instance_vt, type_vt
            )
        return type_attr

    # Step 6: __getattr__ fallback
    getattr_fn = type_vt.lookup_mro("__getattr__")
    if getattr_fn is not None:
        return getattr_fn.ob_type_vt.tp_call(
            tx, getattr_fn, [instance_vt, ConstantVariable.create(name)], {}
        )

    raise ObservedAttributeError(...)
```

### `generic_setattr` — shared default for `tp_setattro`

```python
def generic_setattr(tx, instance_vt, name, value_vt):
    """PyObject_GenericSetAttr."""
    type_vt = instance_vt.ob_type_vt

    # Step 1: Data descriptor intercepts write
    type_attr = type_vt.lookup_mro(name)
    if type_attr is not None:
        descr_set = type_attr.ob_type_vt.tp_descr_set
        if descr_set is not None:
            return descr_set(tx, type_attr, instance_vt, value_vt)

    # Step 2: Write to instance __dict__
    if type_vt.tp_flags.has_instance_dict:
        tx.output.side_effects.store_attr(instance_vt, name, value_vt)
        return CONSTANT_VARIABLE_NONE

    # Step 3: No __dict__, no data descriptor
    raise_observed_exception(AttributeError, tx, ...)
```

## Concrete Examples

### Example: `list` type and `ListVariable`

#### Slot implementations (extracted from current `BaseListVariable.call_method`)

```python
def _list_mp_subscript(tx, self_vt, key_vt):
    """list.__getitem__"""
    if key_vt.python_type() not in (int, slice):
        raise_observed_exception(TypeError, tx, ...)
    return self_vt.getitem_const(tx, key_vt)

def _list_sq_contains(tx, self_vt, value_vt):
    """list.__contains__"""
    return iter_contains(self_vt.unpack_var_sequence(tx), value_vt, tx)

def _list_sq_length(tx, self_vt):
    """list.__len__"""
    return ConstantVariable.create(len(self_vt.items))

def _list_sq_concat(tx, self_vt, other_vt):
    """list.__add__"""
    if type(self_vt) is not type(other_vt):
        raise_observed_exception(TypeError, tx, ...)
    return ListVariable(self_vt.items + other_vt.items)

def _list_tp_iter(tx, self_vt):
    """list.__iter__"""
    return ListIteratorVariable(self_vt.items, mutation_type=ValueMutationNew())

def _list_tp_richcompare(tx, self_vt, other_vt, op):
    """list comparison"""
    if not isinstance(other_vt, BaseListVariable):
        if op == "__eq__": return ConstantVariable.create(False)
        elif op == "__ne__": return ConstantVariable.create(True)
        raise_observed_exception(TypeError, tx, ...)
    return polyfills.list_cmp(op, self_vt, other_vt)

def _list_tp_hash(tx, self_vt):
    """list.__hash__ — list is unhashable."""
    raise_observed_exception(TypeError, tx, args=["unhashable type: 'list'"])

def _list_append(tx, self_vt, value_vt):
    """list.append — in tp_dict, not a slot."""
    tx.output.side_effects.mutation(self_vt)
    self_vt.items.append(value_vt)
    return CONSTANT_VARIABLE_NONE

# ... extend, insert, pop, clear, copy, reverse, remove, sort, index, count
```

#### Type singleton

```python
LIST_TYPE_VT = TypeVariableTracker(
    python_cls=list,
    tp_base=OBJECT_TYPE_VT,
    tp_mro=(LIST_TYPE_VT, OBJECT_TYPE_VT),
    tp_flags=TypeFlags(
        has_instance_dict=False,
        is_immutable_type=True,
        is_sequence=True,
        is_hashable=False,
    ),
    tp_getattro=generic_getattr,
    tp_setattro=generic_setattr,
    tp_iter=_list_tp_iter,
    tp_hash=_list_tp_hash,
    tp_richcompare=_list_tp_richcompare,
    sq_length=_list_sq_length,
    sq_contains=_list_sq_contains,
    sq_concat=_list_sq_concat,
    sq_repeat=_list_sq_repeat,
    sq_inplace_concat=_list_sq_inplace_concat,
    sq_inplace_repeat=_list_sq_inplace_repeat,
    mp_subscript=_list_mp_subscript,
    mp_ass_subscript=_list_ass_item,
    tp_codegen_sync=_list_codegen_sync,   # old[:] = new
    tp_dict={
        "append": _make_builtin_method_vt(_list_append),
        "extend": _make_builtin_method_vt(_list_extend),
        "insert": _make_builtin_method_vt(_list_insert),
        "pop": _make_builtin_method_vt(_list_pop),
        "clear": _make_builtin_method_vt(_list_clear),
        "copy": _make_builtin_method_vt(_list_copy),
        "reverse": _make_builtin_method_vt(_list_reverse),
        "remove": _make_builtin_method_vt(_list_remove),
        "sort": _make_builtin_method_vt(_list_sort),
        "index": _make_builtin_method_vt(_list_index),
        "count": _make_builtin_method_vt(_list_count),
    },
)
```

#### `ListVariable` — data only

```python
class ListVariable(VariableTracker):
    """A list instance. Carries items — nothing else."""
    items: list[VariableTracker]

    def __init__(self, items, **kwargs):
        super().__init__(**kwargs)
        self.ob_type_vt = LIST_TYPE_VT
        self.items = list(items)

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        return [x.as_python_constant() for x in self.items]

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    # No call_method. No var_getattr. No python_type().
```

### Example: list subclass gets correct semantics automatically

```python
class MyList(list):
    def __getitem__(self, key):
        print("custom getitem")
        return super().__getitem__(key)
```

Dynamo constructs:

```python
MYLIST_TYPE_VT = TypeVariableTracker(
    python_cls=MyList,
    tp_base=LIST_TYPE_VT,
    tp_mro=(MYLIST_TYPE_VT, LIST_TYPE_VT, OBJECT_TYPE_VT),
    tp_flags=TypeFlags(has_instance_dict=True, is_heaptype=True, is_sequence=True),

    # Inherits all slots from LIST_TYPE_VT except overridden ones
    tp_getattro=generic_getattr,
    tp_iter=LIST_TYPE_VT.tp_iter,              # inherited
    sq_length=LIST_TYPE_VT.sq_length,          # inherited

    # mp_subscript is NULL — overridden __getitem__ lives in tp_dict
    mp_subscript=None,

    tp_dict={
        "__getitem__": UserFunctionVariable(MyList.__getitem__),
        # everything else inherited via MRO walk
    },
)
```

When `obj[key]` is executed and `mp_subscript` is NULL, dispatch falls through
to `var_getattr("__getitem__")`, which walks the MRO, finds `__getitem__` in
`MYLIST_TYPE_VT.tp_dict`, invokes the descriptor protocol, and calls it.

### Example: functions and methods

#### `FUNCTION_TYPE_VT`

```python
def _function_tp_call(tx, func_vt, args, kwargs):
    """Inline the function."""
    return tx.inline_user_function_return(
        func_vt, [*func_vt.self_args(), *args], kwargs
    )

def _function_tp_descr_get(tx, func_vt, instance_vt, owner_type_vt):
    """
    function.__get__ — creates bound methods.
    Called by generic_getattr when a function is found in the MRO.
    """
    if instance_vt is None:
        return func_vt  # accessed on class itself
    return BoundMethodVariable(func_vt=func_vt, obj_vt=instance_vt)

FUNCTION_TYPE_VT = TypeVariableTracker(
    python_cls=types.FunctionType,
    tp_base=OBJECT_TYPE_VT,
    tp_mro=(FUNCTION_TYPE_VT, OBJECT_TYPE_VT),
    tp_flags=TypeFlags(has_instance_dict=True, is_immutable_type=True),
    tp_call=_function_tp_call,
    tp_descr_get=_function_tp_descr_get,
    tp_descr_set=None,                      # NON-data descriptor
    tp_getattro=_function_tp_getattro,
    tp_setattro=generic_setattr,            # plain generic — functions have __dict__
)
```

#### `UserFunctionVariable` — data only

```python
class UserFunctionVariable(VariableTracker):
    fn: types.FunctionType
    is_constant: bool

    def __init__(self, fn, is_constant=False, **kwargs):
        super().__init__(**kwargs)
        self.ob_type_vt = FUNCTION_TYPE_VT
        self.fn = fn
        self.is_constant = is_constant

    def get_code(self): return self.fn.__code__
    def get_globals(self): return self.fn.__globals__
    def self_args(self): return []
    def as_python_constant(self): return self.fn
    # No var_getattr, no call_function overrides.
```

#### `BoundMethodVariable` — separate type (NOT inheriting from function)

```python
def _method_tp_call(tx, method_vt, args, kwargs):
    """PyMethod_Type.tp_call — call underlying function with self prepended."""
    func_vt = method_vt.func_vt
    obj_vt = method_vt.obj_vt
    return func_vt.ob_type_vt.tp_call(tx, func_vt, [obj_vt, *args], kwargs)

def _method_tp_getattro(tx, method_vt, name):
    """Bound method attribute access."""
    if name == "__self__":
        return method_vt.obj_vt
    if name == "__func__":
        return method_vt.func_vt
    # Delegate most attribute access to underlying function
    return method_vt.func_vt.ob_type_vt.tp_getattro(tx, method_vt.func_vt, name)

METHOD_TYPE_VT = TypeVariableTracker(
    python_cls=types.MethodType,
    tp_base=OBJECT_TYPE_VT,
    tp_mro=(METHOD_TYPE_VT, OBJECT_TYPE_VT),
    tp_flags=TypeFlags(has_instance_dict=False),
    tp_call=_method_tp_call,
    tp_descr_get=None,
    tp_getattro=_method_tp_getattro,
)

class BoundMethodVariable(VariableTracker):
    """NOT a subclass of UserFunctionVariable."""
    func_vt: VariableTracker   # __func__
    obj_vt: VariableTracker    # __self__

    def __init__(self, func_vt, obj_vt, **kwargs):
        super().__init__(**kwargs)
        self.ob_type_vt = METHOD_TYPE_VT
        self.func_vt = func_vt
        self.obj_vt = obj_vt
```

#### How `obj.method(args)` flows end-to-end

```
1. LOAD_ATTR "method" on instance_vt
   → instance_vt.ob_type_vt.tp_getattro(tx, instance_vt, "method")
   → generic_getattr:
       → Step 1: walks MRO, finds func_vt in some class's tp_dict
       → Step 3: not in instance dict
       → Step 4: func_vt.ob_type_vt is FUNCTION_TYPE_VT
                  FUNCTION_TYPE_VT.tp_descr_get is not None → descriptor
                  → _function_tp_descr_get(tx, func_vt, instance_vt, owner_type_vt)
                  → returns BoundMethodVariable(func_vt, instance_vt)

2. CALL on the BoundMethodVariable
   → bound_method_vt.ob_type_vt.tp_call(tx, bound_method_vt, args, kwargs)
   → _method_tp_call:
       → calls func_vt.ob_type_vt.tp_call(tx, func_vt, [obj_vt, *args], kwargs)
       → _function_tp_call:
           → tx.inline_user_function_return(func_vt, [obj_vt, *args], kwargs)
```

Same code path regardless of whether `instance_vt` is a `UserDefinedObjectVariable`,
`TensorVariable`, or `ListVariable`.

#### Other descriptor types

```python
STATICMETHOD_TYPE_VT = TypeVariableTracker(
    python_cls=staticmethod,
    tp_descr_get=lambda tx, sm_vt, instance_vt, owner_vt: sm_vt.wrapped_fn_vt,
    tp_descr_set=None,  # non-data descriptor
    ...
)

CLASSMETHOD_TYPE_VT = TypeVariableTracker(
    python_cls=classmethod,
    tp_descr_get=lambda tx, cm_vt, instance_vt, owner_vt:
        BoundMethodVariable(cm_vt.wrapped_fn_vt, owner_vt),
    tp_descr_set=None,  # non-data descriptor
    ...
)

PROPERTY_TYPE_VT = TypeVariableTracker(
    python_cls=property,
    tp_descr_get=lambda tx, prop_vt, instance_vt, owner_vt:
        prop_vt.fget_vt.ob_type_vt.tp_call(tx, prop_vt.fget_vt, [instance_vt], {}),
    tp_descr_set=lambda tx, prop_vt, instance_vt, value_vt:
        prop_vt.fset_vt.ob_type_vt.tp_call(tx, prop_vt.fset_vt, [instance_vt, value_vt], {}),
    ...  # DATA descriptor — both get and set
)
```

### Example: `func.foo = "bar"`

Functions have `has_instance_dict=True` and no custom `tp_setattro`:

```
STORE_ATTR "foo" on func_vt
→ generic_setattr:
    → Step 1: FUNCTION_TYPE_VT.lookup_mro("foo") → None
    → Step 2: has_instance_dict=True → side_effects.store_attr(func_vt, "foo", bar_vt)
```

At codegen time:

```
codegen_store_attr(cg, func_vt, "foo", bar_vt)
→ FUNCTION_TYPE_VT.tp_setattro is generic_setattr (no custom __setattr__)
→ FUNCTION_TYPE_VT.lookup_mro("foo") → None (no descriptor)
→ emit plain STORE_ATTR
```

### Example: `obj.x = 5` where `x` is a `property` with a setter

```python
class Foo:
    @property
    def x(self): return self._x
    @x.setter
    def x(self, val): self._x = val
```

```
STORE_ATTR "x" on foo_vt
→ generic_setattr:
    → Step 1: FOO_TYPE_VT.lookup_mro("x") → prop_vt
              prop_vt.ob_type_vt is PROPERTY_TYPE_VT
              PROPERTY_TYPE_VT.tp_descr_set is not None → data descriptor
              → PROPERTY_TYPE_VT.tp_descr_set(tx, prop_vt, foo_vt, value_vt)
              → calls prop_vt.fset_vt through tp_call
              → never touches instance dict
```

### Example: `obj.x = 5` shadows a method (non-data descriptor)

```python
class Foo:
    def x(self): pass

foo = Foo()
foo.x = 5  # shadows the method in instance dict
```

```
→ generic_setattr:
    → Step 1: FOO_TYPE_VT.lookup_mro("x") → func_vt
              func_vt.ob_type_vt is FUNCTION_TYPE_VT
              FUNCTION_TYPE_VT.tp_descr_set is None → NOT a data descriptor
              → falls through
    → Step 2: has_instance_dict=True
              → side_effects.store_attr(foo_vt, "x", ConstantVariable(5))
```

After this, `foo.x` returns `5` (instance dict, step 3 of `generic_getattr`),
while `Foo.x` still returns the function.

### Binary operation dispatch

Replaces isinstance cascades in `BuiltinVariable`:

```python
def binary_op(tx, op_slot, v, w):
    """Mirrors CPython's binary_op1."""
    v_slot = getattr(v.ob_type_vt, op_slot, None)
    w_slot = getattr(w.ob_type_vt, op_slot, None)

    # Subclass priority
    if (w.ob_type_vt.is_subtype_of(v.ob_type_vt)
            and w_slot is not v_slot):
        result = w_slot(tx, w, v, reflected=True)
        if result is not NotImplemented:
            return result
        w_slot = None

    if v_slot is not None:
        result = v_slot(tx, v, w)
        if result is not NotImplemented:
            return result

    if w_slot is not None:
        result = w_slot(tx, w, v, reflected=True)
        if result is not NotImplemented:
            return result

    raise TypeError(...)
```

## Reconstruction Simplification

### Current state: ~350-line isinstance cascade in `codegen_update_mutated`

```python
# Current side_effects.py codegen_update_mutated (simplified):
if isinstance(var, variables.ListVariable):
    # old[:] = new
elif isinstance(var, variables.lists.DequeVariable):
    # old.clear(); old.extend(new)
elif isinstance(var, variables.ConstDictVariable):
    # old.clear(); old.update(new)
elif isinstance(var, variables.TorchFunctionModeStackVariable):
    # torch function mode-specific
elif isinstance(var, variables.CellVariable):
    # cell-specific
elif self.is_attribute_mutation(var):
    if isinstance(var, variables.UserDefinedDictVariable):
        # dict subclass-specific
    elif isinstance(var, variables.UserDefinedListVariable):
        # list subclass-specific
    # Then for store_attr_mutations:
    if isinstance(var, variables.NewGlobalVariable):
        # STORE_GLOBAL
    elif isinstance(value, variables.DeletedVariable):
        # DELETE_ATTR
    elif isinstance(var, UDOV) and var.should_skip_descriptor_setter(name):
        # call object_setattr_ignore_descriptor
    elif isinstance(var, UDOV) and var.needs_slow_setattr():
        # call object.__setattr__
    else:
        # STORE_ATTR
elif isinstance(var, variables.ListIteratorVariable):
    # advance the iterator
elif isinstance(var, variables.RandomVariable):
    # set random state
```

### New regime: type-driven, ~20 lines

```python
def codegen_update_mutated(self, cg):
    suffixes = []
    for var in self._get_modified_vars():
        type_vt = var.ob_type_vt

        # Value mutation: delegate to type's sync strategy
        if type_vt.tp_codegen_sync is not None:
            suffix = type_vt.tp_codegen_sync(cg, var)
            if suffix: suffixes.append(suffix)

        # Attribute mutation: uniform for all types
        if var in self.store_attr_mutations:
            for name, value in reversed(self.store_attr_mutations[var].items()):
                suffix = codegen_store_attr(cg, var, name, value)
                if suffix: suffixes.append(suffix)

    for suffix in reversed(suffixes):
        cg.extend_output(suffix)
```

### `codegen_store_attr` — type-driven decisions

```python
def codegen_store_attr(cg, var, name, value):
    type_vt = var.ob_type_vt

    # Custom __setattr__? Bypass it.
    if type_vt.tp_setattro is not generic_setattr:
        # emit object.__setattr__(var, name, value)
        cg.load_import_from("builtins", "object")
        cg.load_method("__setattr__")
        cg(var.source)
        cg(ConstantVariable(name))
        cg(value)
        return [*create_call_method(3), create_instruction("POP_TOP")]

    # Data descriptor traced through?
    type_attr = type_vt.lookup_mro(name)
    if type_attr is not None and type_vt.is_data_descriptor(type_attr):
        descr_set = type_attr.ob_type_vt.tp_descr_set
        if is_python_function(descr_set):
            return None  # traced, no-op
        else:
            # C descriptor, bypass: object_setattr_ignore_descriptor
            cg.add_push_null(
                lambda: cg.load_import_from(utils.__name__, "object_setattr_ignore_descriptor")
            )
            cg(var.source)
            cg(ConstantVariable(name))
            cg(value)
            return [*create_call_function(3, False), create_instruction("POP_TOP")]

    # Plain STORE_ATTR
    cg.tx.output.update_co_names(name)
    cg(value)
    cg(var)
    return [create_instruction("STORE_ATTR", argval=name)]
```

### `codegen_save_tempvars` — type-driven construction

```python
for var in self._get_modified_vars():
    if not isinstance(var.mutation_type, AttributeMutationNew):
        continue
    type_vt = var.ob_type_vt
    type_vt.tp_codegen_new(cg, var)  # each type knows how to construct itself
    cg.add_cache(var)
    var.source = TempLocalSource(cg.tempvars[var])
```

### Summary of reconstruction changes

| Decision | Today (isinstance checks) | New regime (type-driven) |
|---|---|---|
| How to sync a mutated container | `isinstance(var, ListVariable)` vs `DequeVariable` vs `ConstDictVariable` | `type_vt.tp_codegen_sync` |
| How to write an attribute | `should_skip_descriptor_setter` + `needs_slow_setattr` + `isinstance(UDOV)` | `type_vt.tp_setattro` + `type_vt.lookup_mro(name)` |
| How to construct a new object | `isinstance(CellVariable)` vs `is_tensor()` vs `isinstance(UDOV)` | `type_vt.tp_codegen_new` |
| How to delete an attribute | hardcoded `DELETE_ATTR` | `codegen_delete_attr` checks type for `__delattr__` |

## What Changes and What Doesn't

### Changes

- Behavioral methods (`var_getattr`, `call_method`, `call_function`, iteration,
  hashing, comparison) move from VT subclasses to `TypeVariableTracker` slots
- VT subclasses become data carriers (like `PyListObject` just carries `ob_item`)
- `VariableTracker` base gains `ob_type_vt` and uniform dispatch
- `BuiltinVariable` shrinks — binary ops, isinstance, hash, etc. dispatch
  through type slots
- `side_effects.py` codegen uses type metadata instead of isinstance cascades
- `UserMethodVariable` no longer inherits from `UserFunctionVariable`

### Doesn't change

- `Source` / guard machinery
- `SideEffects` tracking (`store_attr_mutations`, etc.)
- FX graph building / proxy machinery
- `as_proxy()` stays on VT subclasses (FX representation, not CPython semantics)
- Reconstruction bytecode emission (`reconstruct()`) stays on VT subclasses

## What NOT to Mirror from CPython

- `ob_refcnt` — Dynamo has its own lifetime model
- `tp_dealloc` / `tp_alloc` / `tp_free` — Dynamo doesn't manage C-level memory
- `tp_as_number` / `tp_as_sequence` / `tp_as_mapping` as sub-structs — flatten
  the slots directly onto `TypeVariableTracker` for simplicity
- `tp_version_tag` — Dynamo uses guards instead of version tags

## Migration Path

### Phase 1: Introduce TypeVariableTracker with `tp_getattro`

- Create `TypeVariableTracker` class and `generic_getattr`
- Have `UserDefinedObjectVariable` use it (it already implements the CPython
  algorithm most faithfully — this is the beachhead)
- Other VTs continue as-is

### Phase 2: Add `ob_type_vt` to base VariableTracker

- Auto-construct `TypeVariableTracker` for existing VTs in `__init__`
- The field exists but dispatch is opt-in (old `var_getattr` overrides still
  work as fallback)

### Phase 3: Migrate one built-in type end-to-end

- Pick `list` — exercises `tp_iter`, `sq_length`, `sq_contains`,
  `mp_subscript`, `tp_richcompare`
- Move `ListVariable.call_method` logic into `LIST_TYPE_VT` slots
- Verify all list tests pass

### Phase 4: Migrate binary op dispatch

- Move `BuiltinVariable.call_*` binary op pattern-matching to `binary_op()`
  using type slots
- Big payoff: correct subclass priority by construction

### Phase 5: Migrate remaining types

- Dict, tuple, tensor, module variables
- Migrate function/method to the correct non-inheritance model
- Migrate reconstruction codegen to type-driven

### Coexistence strategy

Each phase must maintain backward compatibility. A VT that hasn't been migrated
uses its existing `var_getattr`/`call_method` overrides. The base class dispatch
checks `ob_type_vt` slots first and falls back to the subclass override if
the type slot is not populated:

```python
def var_getattr(self, tx, name):
    if self.ob_type_vt is not None and self.ob_type_vt.tp_getattro is not None:
        return self.ob_type_vt.tp_getattro(tx, self, name)
    return self._legacy_var_getattr(tx, name)  # old path
```

## Key Benefits

1. **One attribute lookup implementation** — fixes in `generic_getattr` apply
   to all types
2. **Correct subclass semantics** — `MyList(list)` works via MRO + slot
   inheritance, no cliff into UserDefinedObjectVariable
3. **Correct binary op dispatch** — subclass priority, `NotImplemented`
   fallback, all by construction
4. **Descriptor protocol is uniform** — `function.__get__`, `property.__get__`,
   `staticmethod.__get__`, custom descriptors all go through the same path
5. **Type guards are structural** — `ob_type_vt` is a guarded VT with its own
   source; MRO/tp_dict changes are guardable
6. **Reconstruction simplifies** — codegen queries type metadata instead of
   using isinstance cascades
7. **Future operations are cheap** — new dunder methods become new slots on
   `TypeVariableTracker`, not new methods on every VT subclass

## Open Questions

1. **Performance**: The current system benefits from hardcoded fast paths (e.g.,
   `TensorVariable.var_getattr` knows what `tensor.shape` means without MRO
   walk). How do we preserve this? Options: `tp_getattro` overrides per type,
   cached slot lookups, or special-casing common attributes in `generic_getattr`.

2. **Bootstrapping**: `TypeVariableTracker` for built-in types needs to exist
   before any VTs are created. Lazy singletons? Module-level construction?

3. **Metaclasses**: `type` itself is a type. `TypeVariableTracker` for user
   classes needs a metatype. How deep does the recursion go? (CPython stops
   at `PyType_Type.ob_type = &PyType_Type`.)

4. **`tp_dict` representation**: Should `tp_dict` entries be VTs wrapping real
   Python objects, or something lighter? For immutable built-in types, we know
   `tp_dict` won't change, so we might not need full VTs.

5. **Guard granularity**: When a heap type's `tp_dict` is guarded, do we guard
   the entire dict or individual keys? CPython uses `tp_version_tag` as a
   coarse invalidation signal.

6. **`as_proxy` and FX graph**: VT subclasses still need `as_proxy()` for FX
   graph representation. This doesn't map to anything in CPython. Where does
   it live — on the VT instance (current) or somewhere else?
