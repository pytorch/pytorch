---
name: add-op-overload
description: Checklist for adding a new overload (e.g. `mm.dtype`, `addmm.compute_mode`) to an existing ATen op in native_functions.yaml. Use before the first rebuild when adding an overload or new kwarg variant of an existing operator, so every rebuild-triggering edit lands in one build and packet-level Python registrations (meta, decompositions, Inductor lowerings, DTensor) don't silently capture the new overload.
---

# Adding an overload to an existing ATen op

Editing `native_functions.yaml` regenerates headers that most of the tree includes, so it is
close to a full rebuild; so is editing `ATen/Context.h` (e.g. to add a global setting).
`autocast_mode.h` reaches only about a dozen translation units, but forgetting it still costs
another build, relink and test cycle. Each missed item below costs a rebuild or ships a
silent bug. **Finish all of Part 1 before starting the first build.**

Find the closest precedent first and copy its footprint. For matmul ops, that's the
`out_dtype` overloads (`grep -n out_dtype aten/src/ATen/native/native_functions.yaml`).
`git log --oneline -S'<op>.<overload>' -- aten/src/ATen/native/native_functions.yaml` finds
the commit that added it; `git show --stat <sha>` lists every file it touched.

## Part 1: C++ and codegen (needs a rebuild)

1. **Schema** (`native_functions.yaml`): add the functional overload and its `_out`
   overload, each with a new, unique overload name. A thin wrapper that sets state and calls
   the regular op should use `CompositeExplicitAutograd`, so it stays visible to autograd,
   tracing and export instead of decomposing away.
2. **C++ overload resolution.** The C++ API flattens kwarg-only arguments into positional
   parameters, so the new argument lands where existing callers pass `beta`/`alpha`
   positionally, e.g. `at::addmm_out(result, t, sparse, dense, 0, 1)` in
   `aten/src/ATen/native/sparse/SparseTensorMath.cpp`.
   - `str` (`c10::string_view`): the literal `0` converts to it as a null pointer, so the
     call becomes **ambiguous** and the build fails late.
   - `bool` / `int`: `0` converts by a standard conversion, which beats `Scalar`'s
     constructor, so the call **silently binds to the new overload**.

   Fix: put `cpp_no_default_args: ['beta', 'alpha']` (the trailing defaulted args) on the
   new overloads. Check every in-tree C++ call of the op against the new signature. If the
   op previously had a single C++ overload, code that takes its address (`at::mm` passed
   as a function pointer) also stops compiling; wrap it in a lambda.
3. **Autograd** (`tools/autograd/derivatives.yaml`): a `CompositeExplicitAutograd`
   overload needs its own entry; it does not inherit the base op's. If backward should
   keep the new argument's semantics, call the same overload there and thread the argument
   through the helpers in `torch/csrc/autograd/FunctionsManual.{h,cpp}`.
4. **Autocast lists are per-overload.** If the base op is in `AT_FORALL_LOWER_PRECISION_FP`,
   `AT_FORALL_FP32`, etc. in `aten/src/ATen/autocast_mode.h`, or in the `KERNEL_CPU(...)` /
   `KERNEL_MPS(...)` lists in `aten/src/ATen/autocast_mode.cpp`, add `_(op, overload)` /
   `KERNEL_CPU(op, overload, policy)` with the same policy. Without this, the wrapper's
   inner call is autocast but its autograd node is not: forward runs in low precision and
   backward fails with mixed dtypes (`expected mat1 and mat2 to have the same dtype`).
5. **Other per-overload C++ tables**
   (`grep -rnE '"(aten::)?<op>[."]|VMAP_SUPPORT\(<op>,' aten/src/ATen --include=*.cpp`):
   - vmap batch rules (`VMAP_SUPPORT` / `m.impl` in `aten/src/ATen/functorch/BatchRules*.cpp`).
     Missing means the slow per-example fallback.
   - Conjugate fallthroughs in `aten/src/ATen/ConjugateFallback.cpp`. Missing means conj
     inputs get materialized first.

## Part 2: Python (no rebuild needed)

6. **Packet-level registrations capture the new overload.** Registering on an
   `OpOverloadPacket` (`aten.op` rather than `aten.op.default`) expands to *every* overload,
   including yours, and then silently drops the new argument. Audit:
   ```bash
   grep -rnE "aten\.<op>\b[^.]" torch --include=*.py | grep -E "register|lowering|decomp|fallback|flop"
   ```
   Typical hits: `register_meta(aten.op)` in `torch/_meta_registrations.py`,
   `register_decomposition([aten.op])` in `torch/_decomp/decompositions.py` and
   `torch/_inductor/decomposition.py`, and Inductor `register_lowering(aten.op)` in
   `torch/_inductor/kernel/`. Narrow each one to explicit overloads
   (`aten.op.default, aten.op.out`) unless it really handles the new argument. Adding your
   own registration while a packet one exists fails with `duplicate registrations for ...`.
   `torch/utils/flop_counter.py` looks formulas up by packet, so it can't be narrowed;
   check that the formula's `*args, **kwargs` accept the new overload's arguments.
7. **Inductor**: to run the overload eagerly, add `make_fallback(aten.op.overload)` in
   `torch/_inductor/lowering.py`. It must run before `import_submodule(kernel)` at the end
   of that file, because `get_overloads` only expands a packet to overloads that don't
   have a lowering yet. `make_fallback` asserts if a decomposition still covers the
   overload, so narrow the decompositions first.
8. **DTensor**: sharding strategies are per-overload (e.g.
   `register_single_dim_strategy(aten.mm.default)` in
   `torch/distributed/tensor/_ops/_matrix_ops.py`). Without one, DTensor uses
   `decomposition_table` if it covers the overload (dropping the argument if a packet
   decomposition captured it), and otherwise raises `does not have a sharding strategy
   registered`.
9. **Expect files**: `EXPECTTEST_ACCEPT=1 python test/test_decomp.py -k HasDecomp`. The diff
   should show only the new overloads.
10. **Signatures and docs**: update the `get_testing_overrides()` lambdas in
    `torch/overrides.py` if the Python signature gains a parameter. In
    `torch/_torch_docs.py`, add a `.. function::` block with `:noindex:` for the new
    signature, as the `out_dtype` ones do, plus any relevant `docs/source/notes/*.md`.

## Part 3: Validation checklist for the new overload

- Eager forward and backward, plus `gradcheck` (including forward AD) on CPU.
- `torch.autocast` forward **and backward** on CUDA and CPU (catches item 4).
- FakeTensor/meta shapes; `torch.export` + `run_decompositions()` keeps the overload;
  a `TorchDispatchMode` sees `aten.op.overload` (catches item 6).
- `torch.compile` forward and backward: assert that the generated code *calls* the
  overload with the argument, rather than just counting occurrences of the op name.
- vmap and DTensor, if they apply (items 5 and 8).
- Use **pytest** for `-k "a or b"` expressions. `python test/X.py -k` goes to unittest,
  which treats the expression as a literal substring and runs 0 tests.
