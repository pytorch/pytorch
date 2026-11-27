# Draft-only schema adaptation PoC

This is an executable design fixture above the build/target plumbing in #205828,
not a landable API proposal. **All build, target, and runtime 14/16/17 labels are
SIMULATED metadata and schema fixtures.** The test uses this checkout's actual
2.16 headers and one current libtorch, not real 2.17 headers or a 2.14 binary.
No global ABI getter or compiler version macro is overridden.

## Two schemas, one process

The CPU `test_shim` tests define only private test namespaces with local
`MAKE_TORCH_LIBRARY` objects; their registrations disappear at scope exit:

- `_test_schema_poc_v14::example(Tensor x, int n) -> Tensor`
- `_test_schema_poc_v17::example(Tensor x, int n, bool new_mode=False) -> Tensor`

Both reshape six elements to `{n, -1}`. The new schema transposes that result
when `new_mode=True`; `n=2` and `n=3` expose argument-order/forwarding mistakes.
The integer schema intentionally isolates schema shape from SymInt encodings.
Both endpoints use the existing Tensor-handle, int64_t, and bool StableIValue
encodings; this fixture changes slot shape and metadata, not those encodings.
No real ATen schema, Dummy type, or general torchgen behavior changes.

## Extension calls the dispatcher

Users choose `TORCH_TARGET_VERSION`; without it the effective target defaults
to the build version. `Example<Build, Target>` models future wrapper generation,
with Build standing for `TORCH_ABI_VERSION` and Target for the internal effective
`TORCH_FEATURE_VERSION`. For Target **>= 17**, the public method accepts
`new_mode=False` and boxes three slots. For older targets it accepts only `(x,n)`
and boxes two. Compile-time detection proves the old API cannot accept the new
argument; Target must be nonzero and <= Build. Runtime < Target is rejected
before boxing or invoking any C API.

Injected runtime capabilities choose the schema namespace and API availability.
Build >= 16 plus Runtime >= 16 dynamically resolves and **calls the real
`torch_call_dispatcher_v2`**, even for Target 14. This passes the exact target:
B17/T14/R17 sends two slots with build 17 and target 14. Selecting v2 solely by
Target >= 16 would lose that information. The simulated R14 branch deliberately
uses `torch_call_dispatcher(..., Build)` despite v2 existing in the actual test
library; simulated Build 14 also uses that legacy route. If a build-17,
older-target wrapper needs the two-slot layout on the new schema but v2 is
unavailable, it rejects before boxing: v1 would lose target and incorrectly
select the three-slot build-17 layout.

The private registry keeps historical registrations **build-only**. A separate
`register_targeted_schema_adapter` opts this test key into target-gated lookup:
use a known nonzero target, otherwise build. Target zero is **unknown**, not 14
or universally pre-16: a direct legacy B17 caller still supplies three
build-schema slots, including `new_mode=True`, with no adapter.

The one libtorch-side PoC adapter validates the current `(Tensor,int,bool)`
shape, consumes the old two slots through existing `to_ivalue` conversions with
build/target metadata, and inserts `IValue(false)`. The caller's tensor handle
transfers to libtorch; the returned owning handle transfers back to stable RAII.
There is no automatic adaptation of arbitrary future schemas or return values.

`SimulatedCallMatrix` covers all valid B14/B16/B17, T14/T16/T17, R14/R17 rows,
including B17/T14/R14 and B17/T14/R17, and records C route, slot count, build and
known target. B17/T17/R17 exercises default False and explicit True. Removing
target from lookup breaks the critical two-slot B17/T14/R17 case; treating
legacy target zero as old breaks the direct three-slot B17 case.

## Dispatcher calls an extension kernel

#205828 records build/target for conversions; it does **not** implement generic
registered-kernel schema adapters or bundle C++ type information with callbacks.
`ManualRegisteredKernelBridge` therefore illustrates the extra handwritten work:

- R14 registers `old_cpp_kernel(Tensor,int64_t)` directly with the legacy API.
- R17 registers a three-C++-argument bridge through real dynamically resolved
  `torch_library_impl_v2`, carrying simulated Build 17 / Target 14 metadata.
- The bridge owns its consumed Tensor, rejects True **before the old body**, and
  calls the old two-argument kernel (stable `reshape`) only for False. RAII cleans
  up success and exception paths; the test checks calls and restored refcounts.

The callback remains the existing raw function pointer with three stack/count
parameters. This bridge is not a typed-v3 prototype or a generic kernel adapter.

## Historical anchor and validation

[163832](https://github.com/pytorch/pytorch/pull/163832) changed Dummy's layout for
new-library/old-extension conversions;
[164376](https://github.com/pytorch/pytorch/pull/164376) illustrated the reverse.
[165284](https://github.com/pytorch/pytorch/pull/165284) added a `scale` schema PoC
using fake ABI versions. This fixture follows that concrete illustration style,
without changing value encodings or pretending to run multiple real releases.

After a normal CPU test-enabled PyTorch build, run
`build/bin/test_shim --gtest_filter='SchemaAdaptationPoc.*'`. Passing these tests
would establish this simulated contract only; actual multi-release builds,
public header generation, generic callback adaptation, and publication are
separate work. Keep the PR draft-only for human design review.
