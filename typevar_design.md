# Design: tp_slot Hooks on VariableTracker — Mirroring CPython's Type Slots in Dynamo

**Status**: Active implementation
**Author**: anijain (with Claude)
**Date**: 2026-03-31

## Approach

Instead of introducing a separate `TypeVariableTracker` class, we implement
CPython's tp_slot pattern **directly on each VT subclass** as method hooks.
Each CPython type slot (tp_richcompare, tp_hash, tp_iter, etc.) becomes a
method on the VT subclass that mirrors the semantics of the corresponding C
slot function. The base `VariableTracker` provides the generic CPython
algorithm (e.g., `generic_richcompare` implementing `PyObject_RichCompare`),
and each VT subclass provides its type-specific slot implementation.

This approach was validated by the `richcompare` branch, which successfully
implemented `tp_richcompare` across all VT subclasses with a unified
`generic_richcompare` dispatcher.

## CPython Background

### Type slots

CPython's `PyTypeObject` defines behavior through slot functions — function
pointers on the type struct. When Python executes `a == b`, it doesn't call
`a.__eq__(b)` directly. Instead it calls `PyObject_RichCompare(a, b, Py_EQ)`,
which reads `type(a)->tp_richcompare` and `type(b)->tp_richcompare` and
applies the 4-step algorithm (subclass priority, forward, reflected, fallback).

The same pattern applies to all operations:

| Python syntax | CPython entry point | Slot consulted |
|---|---|---|
| `a == b` | `PyObject_RichCompare` | `tp_richcompare` |
| `a + b` | `binary_op1` | `nb_add` |
| `len(a)` | `PyObject_Size` | `sq_length` / `mp_length` |
| `a[k]` | `PyObject_GetItem` | `mp_subscript` / `sq_item` |
| `hash(a)` | `PyObject_Hash` | `tp_hash` |
| `iter(a)` | `PyObject_GetIter` | `tp_iter` |
| `a.x` | `PyObject_GetAttr` | `tp_getattro` |
| `a()` | `PyObject_Call` | `tp_call` |

Each slot function returns its result or `NotImplemented` to signal that
the type doesn't handle that operation, letting the generic algorithm try
the other operand.

### PyObject_RichCompare — the comparison algorithm

```
1. If type(rhs) is a proper subclass of type(lhs) AND overrides the
   reflected op: try rhs's slot first
2. Try lhs->tp_richcompare(lhs, rhs, op)
3. If NotImplemented, try rhs->tp_richcompare(rhs, lhs, reflected_op)
4. If still NotImplemented:
   - __eq__/__ne__: fall back to identity (a is b)
   - ordering ops: raise TypeError
```

This algorithm is the same for every type. The per-type behavior is isolated
to the `tp_richcompare` slot function, which only needs to handle its own
type and return `NotImplemented` for types it doesn't understand.

## Design Pattern

### The hook method

Each CPython slot becomes a method on the VT subclass. For `tp_richcompare`:

```python
class VariableTracker:
    def richcompare_impl(self, tx, other, op):
        """tp_richcompare slot. Subclasses must override."""
        unimplemented(...)
```

The naming convention is `{slot_name}_impl` to distinguish the per-type slot
from the generic algorithm that calls it.

### The generic algorithm

A standalone function implements the CPython entry-point algorithm, calling
the hook on each operand:

```python
def generic_richcompare(tx, lhs, rhs, op):
    """PyObject_RichCompare — the 4-step algorithm."""
    reflected = reflected_richcompare_op[op]

    # Step 1: subclass priority
    rhs_first = (
        lhs_type is not rhs_type
        and issubclass(rhs_type, lhs_type)
        and type_overrides_richcompare(rhs_type, reflected)
    )

    if rhs_first:
        result = rhs.richcompare_impl(tx, lhs, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 2: forward
    result = lhs.richcompare_impl(tx, rhs, op)
    if not is_richcompare_not_implemented(result):
        return result

    # Step 3: reflected (if not already tried)
    if not rhs_first:
        result = rhs.richcompare_impl(tx, lhs, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 4: fallback
    if op in ("__eq__", "__ne__"):
        identity = vt_identity_compare(lhs, rhs)
        ...
    else:
        raise TypeError(...)
```

### Routing from VariableTracker.call_method

The base `call_method` detects comparison dunder names and routes to
`generic_richcompare` instead of per-VT `call_method` overrides:

```python
def call_method(self, tx, name, args, kwargs):
    ...
    elif name in richcmp_op and not kwargs:
        if len(args) == 1:
            return generic_richcompare(tx, self, args[0], name)
        raise_observed_exception(TypeError, tx, ...)
    ...
```

### Per-type slot implementations

Each VT subclass implements `richcompare_impl` matching its CPython type's
`tp_richcompare` semantics. The slot returns `ConstantVariable(NotImplemented)`
for types it doesn't handle.

**Pattern categories observed in the richcompare implementation:**

1. **Identity-based** (`object_richcompare`): For types whose CPython
   `tp_richcompare` is NULL/inherited from object. Identity for `__eq__/__ne__`,
   `NotImplemented` for ordering. Used by: `BaseUserFunctionVariable`,
   `SkipFunctionVariable`, `FunctoolsPartialVariable`, `NNModuleVariable`,
   `UserDefinedClassVariable`, `PythonModuleVariable`, `TracebackVariable`,
   `ExceptionVariable`.

2. **Python-constant** (`python_constant_richcompare_impl`): For types
   that can be reduced to `as_python_constant()` and compared at trace time.
   Used by: `ConstantVariable`, `EnumVariable`, `MethodWrapperVariable`,
   `GetSetDescriptorVariable`, `OrderedSetClassVariable`.

3. **Container-structural**: Custom comparison logic operating on the VT's
   internal structure. Used by: `BaseListVariable` (delegates to
   `polyfills.list_cmp`), `ConstDictVariable` (delegates to
   `polyfills.dict___eq__`), `SetVariable` (direct set ops on `set_items`),
   `RangeVariable` (structural equality), `SliceVariable` (packs to tuple,
   delegates).

4. **Proxy-based**: Creates FX graph nodes for runtime comparison. Used by:
   `TensorVariable` (creates proxy via `tx.output.create_proxy`),
   `SymNodeVariable` (similar proxy creation).

5. **Delegation**: Wraps an inner VT and delegates. Used by:
   `UserDefinedDictVariable`, `UserDefinedSetVariable`,
   `UserDefinedListVariable`, `UserDefinedTupleVariable`,
   `MappingProxyVariable`, `DictKeysVariable`, `DictItemsVariable`.

6. **Resolution**: `GetAttrVariable` resolves itself to a concrete VT and
   re-enters `generic_richcompare`.

7. **Trace-into**: `UserDefinedObjectVariable` checks if the type has a
   pure-Python comparison method and traces into it. Falls back to
   `object_richcompare` otherwise.

### Helper functions in object_protocol.py

```python
# Shared identity-based comparison (object's tp_richcompare)
def object_richcompare(self, tx, other, op):
    if op not in ("__eq__", "__ne__"):
        return ConstantVariable.create(NotImplemented)
    identity = vt_identity_compare(self, other)
    ...

# Shared constant-fold comparison
def python_constant_richcompare_impl(self, tx, other, op):
    if not other.is_python_constant():
        return ConstantVariable.create(NotImplemented)
    self_val = self.as_python_constant()
    result = getattr(type(self_val), op)(self_val, other.as_python_constant())
    return ConstantVariable.create(result)
```

## What the richcompare branch demonstrated

The `richcompare` branch (commit `9b80a3a0`) implemented this pattern for
`tp_richcompare` across 19 files, touching every VT subclass. Key outcomes:

1. **Removed the old call_method comparison path** from `VariableTracker.call_method`:
   the ~40-line `cmp_name_to_op_mapping` isinstance/constant-fold block was
   replaced by a 10-line dispatch to `generic_richcompare`.

2. **Fixed real bugs** that the old scattered approach caused:
   - `SetVariable` returned `False` for `{1,2} == "foo"` instead of using
     identity fallback (CPython returns `False` via `NotImplemented` →
     identity, but for different reasons).
   - `DictItemsVariable` had the same bug.
   - `ConstDictVariable.__ne__` accessed `.value` which doesn't exist on
     all VTs.
   - `ExceptionVariable.args` was wrapped as `ListVariable` instead of
     `TupleVariable`.

3. **Correct subclass priority**: The generic algorithm handles `rhs_first`
   dispatch automatically. Individual VT slots don't need to think about it.

4. **705 new test cases** in `test/dynamo/test_rich_compare.py` covering
   every VT type, cross-type comparisons, ordering errors, exception
   propagation, and subclass priority.

## Next slots to implement

The same pattern applies to each additional CPython slot. Priority order
based on impact (number of isinstance cascades eliminated and bugs fixed):

### tp_hash → `hash_impl`

Currently: `is_python_hashable()` and `get_python_hash()` on each VT, plus
isinstance checks in `BuiltinVariable` for `hash()`.

New: `hash_impl(self, tx) -> VariableTracker` on each VT, with a
`generic_hash(tx, obj)` entry point mirroring `PyObject_Hash`. Unhashable
types raise `TypeError` from their slot. The base VT `hash_impl` calls
`object.__hash__` (identity-based).

### nb_bool → `bool_impl`

Currently: `as_python_constant()` fallback, per-VT `__bool__` in
`call_method`, isinstance checks in `BuiltinVariable`.

New: `bool_impl(self, tx) -> VariableTracker` on each VT, with
`generic_bool(tx, obj)` mirroring `PyObject_IsTrue`. Containers return
`len(self) != 0`. Objects without `__bool__` or `__len__` return `True`.

### tp_iter → `iter_impl`

Currently: per-VT `call_method("__iter__")`, isinstance checks in
`BuiltinVariable` for `iter()`.

New: `iter_impl(self, tx) -> VariableTracker` on each VT, with
`generic_iter(tx, obj)` mirroring `PyObject_GetIter`.

### sq_length / mp_length → `len_impl`

Currently: per-VT `call_method("__len__")`, isinstance checks in
`BuiltinVariable` for `len()`.

New: `len_impl(self, tx) -> VariableTracker`.

### nb_add and binary ops → `binary_op_impl`

Currently: large isinstance cascade in `BuiltinVariable.call_*` for
`operator.add`, `operator.mul`, etc.

New: `binary_op_impl(self, tx, other, op) -> VariableTracker` per VT,
with `generic_binary_op(tx, lhs, rhs, op)` implementing `binary_op1`
(subclass priority, forward, reflected, TypeError fallback).

### tp_getattro → `getattr_impl`

Currently: per-VT `var_getattr` overrides, each reimplementing subsets of
`PyObject_GenericGetAttr`.

New: `getattr_impl(self, tx, name) -> VariableTracker` with
`generic_getattr(tx, obj, name)` implementing the full 6-step algorithm.
This is the largest migration but also the highest payoff.

### tp_call → `call_impl`

Currently: per-VT `call_function` overrides.

New: `call_impl(self, tx, args, kwargs) -> VariableTracker` with
`generic_call(tx, obj, args, kwargs)` mirroring `PyObject_Call`.

## Migration strategy

Each slot is migrated independently following the pattern established by
`richcompare`:

1. Add `{slot}_impl` method to `VariableTracker` base with a default that
   calls `unimplemented()`.

2. Add the `generic_{slot}` algorithm to `object_protocol.py`.

3. Route the relevant `call_method` / `BuiltinVariable` dispatch through
   the generic algorithm.

4. Implement `{slot}_impl` on every VT subclass that needs it.

5. Add comprehensive tests covering every VT type, cross-type interactions,
   error propagation, and subclass priority.

6. Remove the old dispatch code from `call_method` / `BuiltinVariable`.

Each step is a self-contained PR that can land independently. The old and
new paths can coexist during migration: the base `call_method` checks
whether the slot is implemented and falls back to the old path if not.

## Key design decisions

### Why hooks on VT subclasses, not a separate TypeVariableTracker?

The richcompare implementation showed that per-VT hook methods work well
and are simpler to implement incrementally. Each VT subclass already knows
its CPython type's semantics — the hook just extracts that knowledge into a
standardized method signature. No new class hierarchy, no bootstrapping
problems, no metaclass recursion.

A `TypeVariableTracker` may still be valuable later for modeling MRO,
`tp_dict`, and the descriptor protocol, but it's not needed to get the
benefits of unified dispatch algorithms. The slot hooks are a prerequisite
regardless: even if we later add `TypeVariableTracker`, the slot
implementations would be the same functions, just registered on a type
object instead of defined as methods.

### Why `NotImplemented` return, not exceptions?

Matching CPython: slot functions return `NotImplemented` to signal "I don't
handle this." The generic algorithm uses `NotImplemented` to decide whether
to try the other operand or fall back. This is simpler than exception-based
signaling and matches the mental model of anyone reading CPython source.

### What goes in object_protocol.py?

`object_protocol.py` contains:
- The generic algorithms (`generic_richcompare`, future `generic_hash`, etc.)
- Shared slot implementations (`object_richcompare`,
  `python_constant_richcompare_impl`)
- Helper functions (`vt_identity_compare`, `is_richcompare_not_implemented`)

Per-type slot implementations stay in their respective VT files (e.g.,
`BaseListVariable.richcompare_impl` stays in `lists.py`).
