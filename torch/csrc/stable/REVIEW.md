# Reviewing high-level `torch::stable` wrappers

High-level APIs in this directory are convenience wrappers around existing PyTorch
stable ABIs for better UX. They are not a second implementation of the ABI. Review
them by first establishing the exact unstable C++ contract, then checking that
the stable wrapper forwards that contract through the correct ABI boundary, and
finally testing only the parts that the wrapper could have broken.

The stable wrapper should be deliberately small. Do not use these changes to
redesign the API, add convenience behavior, or expand the underlying operator's
correctness coverage.

## Establish the contract

Find the canonical unstable C++ declaration and, for an operator, its dispatcher
schema. Do not infer the contract from the Python API or from one existing call
site.

Compare the stable wrapper with the complete unstable surface:

- Preserve the name, overload set, parameter order, defaults, return type, and
  method or free-function form.
- Preserve reference and `const` qualifiers after substituting stable types such
  as `torch::stable::Tensor` and header-only array references.
- Preserve mutation, aliasing, ownership, lifetime, error, device, and stream
  behavior.
- Preserve every supported call form. A missing overload can silently bind
  positional arguments to a different overload instead of producing a compile
  error.
- Use overloads when the unstable API uses overloads; do not invent a renamed
  stable-only operation to distinguish them.

If exact parity is not possible, the change must identify the concrete
difference and why it is necessary. Treat an unexplained difference as a bug in
the wrapper.

## Keep the bridge mechanical

Delegate to the existing C shim or to the exact dispatcher operator and
overload. Do not reimplement the operation, normalize its inputs, add validation,
or supply behavior that the unstable API does not have.

For dispatcher-backed wrappers, use the schema as the source of truth and check:

- the operator and overload names;
- the number and order of stack arguments, including optional and defaulted
  arguments;
- each stable-to-IValue conversion; and
- the number, order, and conversion of returned values.

Each stable overload must invoke the same dispatcher overload as its unstable
counterpart. Do not delegate between stable overloads if that changes dispatcher
selection. Conversely, do not add a new C shim when an existing stable bridge
already expresses the exact contract. If an existing C shim does not suffice and
a new one is added, the difference should be documented.

Public declarations must use stable or header-only types and headers. They must
not expose ATen or other unstable implementation types. Route C shim failures
through `STABLE_TORCH_ERROR_CODE_CHECK` NOT the outdated `TORCH_ERROR_CODE_CHECK`.

For handles and callbacks, verify the ownership contract explicitly: borrowed
versus owned handles, shared value semantics, who releases the resource, and
whether a callback runs exactly once. Match the unstable type's RAII behavior
rather than creating a new lifetime model, and derive inspiration from how
lifetime is handled by existing stable IValues.

Keep the patch limited to the wrapper, its required ABI plumbing, and focused
tests. Unrelated refactors and speculative convenience APIs make compatibility
changes harder to audit.

## Check the version boundary

Build-time availability and the minimum compatible runtime are different. Base
the `TORCH_FEATURE_VERSION` guard on the oldest runtime whose shim or dispatcher
contract can support the wrapper, not simply on the version in which the C++
wrapper is added.

Verify that the unstable declaration or dispatcher schema has not changed over
the runtime range the wrapper promises to support. New C shims must be recorded
and guarded according to the stable shim versioning rules. Put the test in the
lowest `libtorch_agn` version target that the wrapper claims to support.

## Test the wrapper boundary

Add focused coverage under
`test/cpp_extensions/libtorch_agn_<version>_extension/` and exercise it through
`test/cpp_extensions/test_libtorch_agnostic.py`. The test should compile the
stable declaration with the intended `TORCH_TARGET_VERSION` and call the
`torch::stable` API itself.

These tests prove that the shim preserves access to the API; they do not need to
re-prove the underlying operator. One representative end-to-end case is enough
for a mechanical wrapper. Add another case only for a distinct wrapper code path
or a contract property that the wrapper itself could plausibly break. For example,
separately test dispatcher overloads implemented by different forwarding paths.
Test aliasing, ownership, or stream behavior only when the wrapper manages that
behavior.

Choose values that would expose the forwarding mistake. For example, a negative
divisor can distinguish floor division from truncating division, and a
non-default stream can show that a native handle was not hard-coded. Do not ask
for exhaustive dtype, shape, device, numerical-edge, or error-case matrices
unless the stable wrapper itself branches on those cases.

Before approving, be able to state:

1. which unstable declaration or schema defines the contract;
2. which shim or dispatcher overload implements the bridge;
3. any intentional difference from the unstable API;
4. the oldest supported runtime; and
5. which test demonstrates each compatibility-sensitive seam introduced by the
   wrapper.
