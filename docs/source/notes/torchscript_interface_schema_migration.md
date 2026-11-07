(torchscript-interface-schema-migration)=

# Migrating TorchScript C++ interface method schemas

```{warning}
This is a source- and ABI-breaking change to `c10::InterfaceType` in the installed
`ATen/core/jit_type.h` header. Update affected C++ callers and rebuild binaries
and extensions that depend on these APIs. The old borrowed-schema signatures
are not retained as compatibility shims.
```

## Accessor replacements

Interface method schemas now have an owning receiver when returned to callers.
This removes the interface's internal receiver ownership cycle while keeping
returned schemas valid independently of the caller's interface reference.

| Previous C++ API | Replacement | Result |
| --- | --- | --- |
| `getMethod(name)` | `getMethodSchema(name)` | `std::unique_ptr<FunctionSchema>`; `nullptr` if absent |
| `methods()` | `methodSchemas()` | `std::vector<FunctionSchema>` returned by value |
| `addMethod(FunctionSchema)` | `addMethod(const FunctionSchema&)` | Receiver validation described below |

Keep the returned owner alive while using a schema. For example, given a
shared-owned interface `iface`:

```cpp
auto forward = iface->getMethodSchema("forward");
auto schemas = iface->methodSchemas();
```

Each returned schema owns its actual interface receiver. The values are
snapshots, not references into the interface's private method vector. Code that
previously retained a borrowed pointer or vector reference must retain the new
owning result instead; changing only a method name is not sufficient for all
callers.

For existence or name-only queries, use `hasMethod(name)` or `getMethodNames()`.
These do not materialize method schemas.

## Shared ownership

Returning a method schema requires shared ownership of the interface. Prefer
interfaces created with `InterfaceType::create(...)`. An interface created by
copy construction with `std::make_shared<InterfaceType>(...)` is also
shared-owned and returned schemas use the copy's receiver identity.

A by-value interface copy can still be used for name/existence queries,
subtype/equality checks and subtype diagnostics. Calling `getMethodSchema()` for
an existing method, or `methodSchemas()` for a nonempty interface, on an
unshared copy raises `c10::Error` explaining the shared-ownership requirement.
A missing-method lookup still returns `nullptr`, and an empty interface still
returns an empty schema vector without requiring shared ownership.

## Registering a method

`addMethod` requires at least one argument. The first argument is the receiver:
both its `type()` and `real_type()` must identify the interface receiving the
method. A schema with an `Any` receiver, a sibling interface or a subtype
interface is not accepted as that receiver. Invalid schemas raise `c10::Error`
before insertion.

When constructing a schema for `iface`, use `Argument("self", iface)` for the
receiver, or supply both receiver type fields as `iface` explicitly. The stored
private schema erases this receiver internally; callers must not pre-erase it.

This migration concerns the C++ interface-schema APIs. It does not change the
serialized TorchScript model format or rename the Python interface-schema
binding's `getMethod` method.
