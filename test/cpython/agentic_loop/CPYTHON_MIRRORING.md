# Dynamo CPython Protocol Mirroring

This is orientation material for agents and reviewers working on Dynamo support
for Python object semantics, especially `tp_*`, `nb_*`, `sq_*`, and `mp_*`
protocol behavior.

The goal is to close CPython protocol gaps in a systematic way. The goal is not
to build a second CPython runtime inside Dynamo. We use CPython's algorithms as
the semantic reference, then express them with Dynamo's existing
`VariableTracker` machinery.

## Why this exists

Dynamo historically supported Python behavior by adding logic to individual
`VariableTracker` subclasses and to ad hoc dispatch sites. That works for local
graph breaks, but it creates a fragmented model:

- one CPython algorithm is reimplemented differently across multiple trackers;
- a builtin type can support one operation but miss a closely related one;
- parallel protocol systems can diverge, such as internal equality/hash helpers
  versus user-visible dunder dispatch;
- Python-level constructs can get custom trackers to work around missing object
  protocol primitives;
- agents and reviewers cannot easily tell whether a change fixes the model or
  only patches the latest symptom.

For users, these all show up as graph breaks, confusing exception behavior, or
surprising gaps between eager Python and `torch.compile`.

The direction is to move from "implementation by accident" toward shared,
auditable implementations of CPython object protocol algorithms.

## Current approach

The active design is lightweight:

- put each cross-type CPython protocol algorithm in a shared free function,
  usually in `torch/_dynamo/variables/object_protocol.py`;
- give `VariableTracker` a base hook method for the per-type operation;
- let individual `VariableTracker` subclasses override only the hook body for
  the type-specific behavior;
- route bytecode handlers, builtin handlers, and `call_method` compatibility
  paths through the shared generic function;
- use CPython slot detection where Dynamo needs to know whether a type actually
  has a slot.

For example, a generic operation should look like this shape:

```python
def generic_operation(tx, obj, *args):
    obj_type = maybe_get_python_type(obj)
    if type_implements_relevant_slot(obj_type):
        return obj.relevant_slot_impl(tx, *args)
    raise_observed_exception(TypeError, tx, args=[...])
```

The exact structure differs by protocol, but the split is the same:

- `generic_*` function: CPython dispatch order, fallback order, and user-visible
  exception behavior;
- base `VariableTracker` hook: default behavior for absent or unsupported
  operations;
- subclass hook override: the actual behavior for list, dict, tensor,
  user-defined object, constant, etc.

This keeps the CPython algorithm in one place without forcing a broader
redesign.

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

## CPython model to use as reference

When implementing a protocol, start from the CPython entry point:

- `PyObject_GetItem` for `obj[key]`;
- `PyObject_GetIter` for `iter(obj)`;
- `PyIter_Next` for `next(obj)`;
- `PyObject_IsTrue` for `bool(obj)`;
- `PyObject_Hash` for `hash(obj)`;
- `PyNumber_Index`, `PyNumber_Long`, and `PyNumber_Float` for numeric
  conversion;
- `PyObject_GenericGetAttr` and `PyObject_GenericSetAttr` for attribute access;
- `binary_op1` / `binary_op` in `abstract.c` for binary operators.

CPython uses slots such as `tp_iter`, `tp_hash`, `nb_bool`, `sq_contains`, and
`mp_subscript` to select behavior. In Dynamo, these slots usually map to:

- a slot-detection helper in `object_protocol.py`, backed by
  `torch._C._dynamo.get_type_slots` when needed;
- a base hook on `VariableTracker`, such as `tp_iter_impl`,
  `tp_iternext_impl`, `mp_subscript_impl`, `sq_item_impl`, `sq_contains`,
  `nb_index_impl`, `nb_int_impl`, `nb_float_impl`, or `hash_impl`;
- per-type overrides on the relevant `VariableTracker` subclasses.

## Current local anchors

- `torch/_dynamo/variables/object_protocol.py`: shared CPython-style object
  protocol operations, including `generic_len`, `generic_bool`, `vt_getitem`,
  `generic_getiter`, `generic_iternext`, binary op helpers, `generic_hash`, and
  `generic_contains`.
- `torch/_dynamo/variables/base.py`: base `VariableTracker` hooks for protocol
  bodies.
- `torch/_dynamo/variables/user_defined.py`: user-defined object, class,
  descriptor, MRO, and attribute behavior. Some of this should move toward
  shared generic helpers over time.
- `torch/_dynamo/variables/builtin.py`: legacy builtin dispatch paths that
  should gradually shrink as protocol helpers grow.
- `torch/csrc/dynamo/init.cpp`: CPython slot detection exported from the C
  extension.
- `test/dynamo/test_tp_slots.py`: slot-detection tests.
- `test/dynamo/test_getitem.py`, `test/dynamo/test_nb_bool.py`,
  `test/dynamo/test_nb_index.py`, `test/dynamo/test_nb_float.py`,
  `test/dynamo/test_nb_int.py`, `test/dynamo/test_tp_hash.py`, and
  `test/dynamo/test_contains_protocol.py`: examples of protocol-focused tests.
- `test/dynamo/cpython/3_13/`: broad CPython behavior tests adapted for Dynamo.

## Implementation principles

Use CPython as the semantic spec.

Read the CPython function for the operation and mirror its observable behavior:
slot selection, fallback order, `NotImplemented` handling, subclass priority,
and exception type/message when those are user-visible.

Model the generic algorithm once.

If an operation is a CPython object protocol operation, it usually belongs in
`object_protocol.py` or a nearby shared protocol helper. Per-type
`VariableTracker` code should implement the slot body, not duplicate the slot
selection algorithm.

Separate "slot absent" from "slot present but not implemented in Dynamo".

If CPython would say the operation is invalid for the type, raise the observed
Python exception with CPython-like wording. If CPython says the type has the
slot but Dynamo has no implementation for that slot body yet, use
`unimplemented()` with a message that names the missing slot and points at a
supportable Dynamo gap.

Do not add a special tracker for a Python-level construct when a CPython
primitive is missing.

Custom trackers for things like Python enums, frozen dataclasses, or descriptor
special cases can hide the real gap. Prefer implementing the missing protocol
primitive, such as attribute lookup, descriptor dispatch, metaclass behavior,
hashing, iteration, or getitem.

Avoid large `isinstance` cascades for semantic dispatch.

Some `isinstance` checks are still necessary in the transitional code, but a new
protocol implementation should not spread the same operation across unrelated
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

## Adding a protocol or slot

Use this workflow for new protocol support.

1. Identify the CPython operation and source files.

   Find the public entry point and the slot it dispatches to. Record the
   relevant CPython function in a comment or PR description when it clarifies
   the implementation.

2. Check whether a shared generic function already exists.

   If it exists, route the new case through it. If it does not exist, add a
   generic helper before adding per-type bodies.

3. Check slot detection.

   If Dynamo needs to know whether a Python type implements the slot, make sure
   `torch/csrc/dynamo/init.cpp` exposes the bit and add coverage in
   `test/dynamo/test_tp_slots.py`.

4. Add or update the base `VariableTracker` hook.

   Use a clear hook name matching the protocol, such as `*_impl`. The base
   method should make the absent-slot versus unsupported-slot distinction clear.

5. Implement per-type hook bodies.

   Keep these local and narrow. For example, list indexing belongs with list
   behavior, dict lookup with dict behavior, tensor behavior with tensor
   behavior. The per-type body should not duplicate the generic dispatch order.

6. Route existing call sites through the generic operation.

   Builtins, bytecode handlers, and dunder-string paths in `call_method` should
   call the shared protocol helper where possible. Compatibility paths are
   acceptable during migration, but new behavior should move toward the generic
   helper.

7. Test the behavior as a protocol, not only as a reproducer.

   Cover builtin types, user-defined classes, subclasses of builtin types,
   missing-slot errors, and CPython-defined fallback behavior. Prefer
   `fullgraph=True` when the test is meant to prove there is no graph break.

## Review checklist

Reviewers should ask these questions before accepting a protocol-related
change.

- Does the PR name the CPython operation or slot it is mirroring?
- Is the CPython algorithm implemented through a shared generic path rather
  than a local spot fix?
- Does the per-type code implement only the hook body?
- Does slot detection match CPython for the relevant builtin types and
  subclasses?
- Are "slot absent" and "slot present but unsupported by Dynamo" handled
  differently?
- Do exceptions and messages match CPython where users can observe them?
- Are descriptors, MRO lookup, subclass priority, reflected operations,
  `NotImplemented`, and fallback behavior considered when relevant?
- Does the change reduce or avoid `VariableTracker` overreach for Python-level
  constructs?
- Are guards and side effects installed at the point where Dynamo relies on a
  Python value, type, or mutable state?
- Are tests broad enough that the same class of graph break will not reappear
  under a nearby type or subclass?
- If the PR adds an unavoidable special case, does it explain why the CPython
  primitive cannot be modeled yet?

The best slot PRs make the next nearby slot easier to implement. They should
leave the protocol surface more centralized than they found it.
