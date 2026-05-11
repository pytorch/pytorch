# Dynamo CPython Mirroring

This is orientation material for agents and reviewers working on Dynamo support
for Python object semantics, especially new `tp_*`, `nb_*`, `sq_*`, and `mp_*`
slots.

The goal is not to make Dynamo a second CPython runtime. The goal is to make the
parts of Dynamo that model Python object behavior follow CPython's structure
closely enough that fixes are systematic, auditable, and reusable across types.

## Why this exists

Dynamo historically supported Python behavior by adding logic to individual
`VariableTracker` subclasses and to ad hoc dispatch sites. That works for local
graph breaks, but it creates a fragmented model:

- one CPython algorithm is reimplemented differently across multiple trackers;
- a builtin type can support one operation but miss a closely related one;
- Python-level types get custom trackers to work around missing object-model
  primitives;
- agents and reviewers cannot easily tell whether a change fixes the model or
  only patches the latest symptom.

For users, these all show up as graph breaks, confusing exception behavior, or
surprising gaps between eager Python and `torch.compile`.

The direction is to move from "implementation by accident" toward an explicit
mirror of CPython's object protocol.

## CPython model to mirror

In CPython, every object starts with a pointer to its type:

```c
typedef struct _object {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} PyObject;
```

The type object, `PyTypeObject`, carries the behavior:

- core slots such as `tp_call`, `tp_getattro`, `tp_setattro`, `tp_hash`,
  `tp_iter`, `tp_iternext`, and `tp_richcompare`;
- number slots under `tp_as_number`, such as `nb_bool`, `nb_index`, `nb_int`,
  and binary arithmetic slots;
- sequence slots under `tp_as_sequence`, such as `sq_length`, `sq_item`, and
  `sq_contains`;
- mapping slots under `tp_as_mapping`, such as `mp_length` and
  `mp_subscript`;
- descriptor slots such as `tp_descr_get` and `tp_descr_set`;
- type metadata such as MRO, base type, type dict, flags, and dict offset.

The important design point is that behavior is dispatched through the object's
type. Dynamo should preserve that shape even when the implementation is still
expressed as `VariableTracker` methods.

## Current Dynamo shape

Dynamo is in a transitional state. The long-term direction is a clearer
separation between instance payload and type behavior, but much of today's code
still uses methods on `VariableTracker` subclasses.

The current CPython-oriented pieces are:

- `torch/_dynamo/variables/object_protocol.py`: shared implementations of
  CPython-style generic operations, such as `generic_len`, `generic_bool`,
  `vt_getitem`, `generic_getiter`, `generic_iternext`, and number dispatch;
- `torch/_dynamo/variables/base.py`: base `VariableTracker` hooks such as
  `tp_iter_impl`, `tp_iternext_impl`, `mp_subscript_impl`, `sq_item_impl`,
  `sq_contains`, and `nb_*_impl`;
- `torch/csrc/dynamo/init.cpp`: exposes CPython slot detection via
  `get_type_slots`, `has_slot`, and the `PySequenceSlots`, `PyMappingSlots`,
  `PyNumberSlots`, and `PyTypeSlots` enums;
- `test/dynamo/test_tp_slots.py`: tests that slot detection matches the
  CPython-level expectations Dynamo depends on;
- `test/dynamo/cpython/3_13/`: adapted CPython tests that validate this work
  against broad Python behavior, not just local reproductions.

When adding a new slot, fit into this structure before adding a one-off branch.

## Long-term design direction

The eventual shape is:

- `VariableTracker` instances primarily carry object payload: list items, dict
  entries, tensor proxy, wrapped Python value, source, and mutation state.
- Type behavior lives in a CPython-like type description: slot table, MRO,
  type dict, descriptor behavior, and type flags.
- Generic algorithms live once and dispatch through slots.
- Type-specific implementations are small slot bodies.
- Dynamo-specific behavior, such as bytecode reconstruction and FX proxy
  representation, remains on Dynamo abstractions. It does not need a CPython
  counterpart.

This does not require mirroring irrelevant CPython slots such as memory
management, GC traversal, refcounting, or deallocation. It also does not require
copying CPython's exact C data structures. Mirroring the dispatch shape is the
important part.

## Implementation principles

Use CPython as the spec.

Start from the CPython entry point for the operation. Examples:

- `PyObject_GetItem` for `obj[key]`;
- `PyObject_GetIter` for `iter(obj)`;
- `PyIter_Next` for `next(obj)`;
- `PyObject_IsTrue` for `bool(obj)`;
- `PyObject_Hash` for `hash(obj)`;
- `PyNumber_Index`, `PyNumber_Long`, and `PyNumber_Float` for numeric
  conversion;
- `PyObject_GenericGetAttr` and `PyObject_GenericSetAttr` for attribute access;
- `binary_op1` / `binary_op` in `abstract.c` for binary operators.

Model the generic algorithm once.

If an operation is a CPython object protocol operation, it usually belongs in
`object_protocol.py` or another shared object-protocol helper. The shared code
should decide which slot applies and what CPython does when the slot is absent.
Per-type `VariableTracker` code should implement the slot body.

Separate "slot absent" from "slot present but not implemented in Dynamo".

If CPython would say the operation is invalid for the type, raise the observed
Python exception with CPython-like wording. If CPython says the type has the
slot but Dynamo has no implementation for that slot body yet, use an
`unimplemented` graph break that names the missing slot and points at a
supportable Dynamo gap.

Do not add a special tracker for a Python-level construct when a CPython
primitive is missing.

Custom trackers for things like Python enums, frozen dataclasses, or descriptor
special cases can hide the real gap. Prefer implementing the missing protocol
primitive, such as attribute lookup, descriptor dispatch, metaclass behavior,
hashing, iteration, or getitem.

Avoid large `isinstance` cascades for semantic dispatch.

Dispatch should be driven by the object's Python type and the relevant slot.
Some `isinstance` checks are still necessary in the transitional code, but a new
slot implementation should not spread the operation across unrelated
`BuiltinVariable`, `UserDefinedObjectVariable`, and container-specific branches.

Respect subclass behavior.

CPython slot dispatch has important subclass and reflected-operation rules.
When handling user-defined subclasses of builtin types, check whether CPython
uses an inherited C slot, a generated slot wrapper, a Python dunder method, or a
fallback path. Do not assume exact builtin behavior applies to subclasses.

Respect descriptor ordering for attributes.

For generic attribute access, the relevant CPython order is:

1. data descriptor on the type or MRO;
2. instance dict;
3. non-data descriptor on the type or MRO;
4. plain class attribute;
5. `__getattr__` fallback;
6. `AttributeError`.

For setting attributes, data descriptors intercept first, then instance dict,
then `AttributeError` if neither applies. Non-data descriptors do not intercept
assignment.

Keep Dynamo-specific mechanisms explicit.

Guards, sources, side effects, mutation tracking, bytecode reconstruction, and
FX proxy construction are Dynamo mechanisms. They should be integrated with the
CPython-shaped semantics, but they are not themselves CPython slots.

## Adding a new slot

Use this workflow for new protocol support.

1. Identify the CPython operation and source files.

   Find the public entry point and the slot it dispatches to. Record the
   relevant CPython function in a comment or PR description when it clarifies
   the implementation.

2. Check slot detection.

   If Dynamo needs to know whether a type implements the slot, make sure
   `torch/csrc/dynamo/init.cpp` exposes the bit and add coverage in
   `test/dynamo/test_tp_slots.py`.

3. Add or update the generic algorithm.

   Put cross-type dispatch in `object_protocol.py` or the nearest existing
   shared protocol helper. This layer should encode CPython fallback order and
   exception behavior.

4. Add the base `VariableTracker` hook.

   If the slot needs a per-type body, add a clearly named `*_impl` method to
   `VariableTracker`. The base method should either raise the CPython exception
   for an absent slot or graph break for a present slot that Dynamo has not
   implemented.

5. Implement the per-type slot bodies.

   Keep these local and narrow. For example, list indexing belongs with list
   behavior, dict lookup with dict behavior, tensor behavior with tensor
   behavior. The per-type body should not duplicate the generic slot selection
   algorithm.

6. Route existing call sites through the generic operation.

   Builtins and bytecode handlers should call the generic protocol helper
   rather than hand-rolling the same dunder lookup. Temporary compatibility
   paths in `call_method` are acceptable during migration, but new behavior
   should move toward slot dispatch.

7. Test the behavior as a protocol, not only as a reproducer.

   Cover builtin types, user-defined classes, subclasses of builtin types,
   missing-slot errors, and any fallback behavior CPython defines. Prefer
   `fullgraph=True` when the test is meant to prove there is no graph break.

## Review checklist

Reviewers should ask these questions before accepting a slot-related change.

- Does the PR name the CPython operation or slot it is mirroring?
- Is the operation implemented through a shared protocol path rather than a
  local spot fix?
- Does slot detection match CPython for the relevant builtin types and
  subclasses?
- Are "slot absent" and "slot present but unsupported by Dynamo" handled
  differently?
- Do exceptions and messages match CPython where users can observe them?
- Are descriptors, MRO lookup, subclass priority, reflected operations, and
  fallback behavior considered when relevant?
- Does the change reduce or avoid `VariableTracker` overreach for Python-level
  constructs?
- Are guards and side effects installed at the point where Dynamo relies on a
  Python value, type, or mutable state?
- Are tests broad enough that the same class of graph break will not reappear
  under a nearby type or subclass?
- If the PR adds an unavoidable special case, does it explain why the CPython
  primitive cannot be modeled yet?

## What not to mirror

Do not spend Dynamo complexity on CPython implementation details that do not
affect tracing semantics:

- refcounting and object lifetime slots;
- allocation and deallocation slots;
- GC traversal slots;
- exact C struct layout;
- CPython's internal cache/version-tag machinery when Dynamo guards already
  cover the invariant.

The aim is a predictable semantic model, not a byte-for-byte copy of CPython.

## Useful local anchors

- `torch/_dynamo/variables/object_protocol.py`: generic CPython-style object
  protocol operations.
- `torch/_dynamo/variables/base.py`: base slot hooks on `VariableTracker`.
- `torch/_dynamo/variables/user_defined.py`: user-defined object, class,
  descriptor, MRO, and attribute behavior.
- `torch/_dynamo/variables/builtin.py`: many legacy builtin dispatch paths that
  should gradually shrink as protocol helpers grow.
- `torch/_dynamo/side_effects.py`: mutation replay and reconstruction, which
  should become more type-driven over time.
- `torch/csrc/dynamo/init.cpp`: slot detection exported from the C extension.
- `test/dynamo/test_tp_slots.py`: slot-detection tests.
- `test/dynamo/test_getitem.py`, `test/dynamo/test_nb_bool.py`,
  `test/dynamo/test_nb_index.py`, `test/dynamo/test_tp_hash.py`: examples of
  protocol-focused tests.
- `test/dynamo/cpython/3_13/`: broad CPython behavior tests adapted for Dynamo.
