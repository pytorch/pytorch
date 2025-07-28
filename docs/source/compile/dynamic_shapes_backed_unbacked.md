# Backed vs Unbacked Symints

Backed `SymInts` are symbolic integers that have a concrete value or "hint"
associated with them. This means that torch can use these values to make
decisions about control flow, such as determining which branch of code
to execute. They are typically derived from operations where the size or
value is known or can be inferred.

Unbacked `SymInts` are symbolic integers that do not have a concrete value or
hint. They often arise from data-dependent operations, such as `.nonzero()`
or `.item()`, where the size or value cannot be determined at compile time.
Since they lack a concrete value, they cannot be used for control flow
decisions, and attempting to do so requires a graph break.

In summary, backed `SymInts` have known values that can be used for
decision-making, while unbacked `SymInts` do not, requiring special handling
to avoid graph breaks.

Unbacked symbolic integers can be too restrictive, causing most PyTorch programs
to fail. To address this, you can use the following methods and APIs as
workaround:

* Use higher-level APIs like `empty` instead of `empty_strided` to create tensors.
This ensures the tensor is non-overlapping and dense, avoiding unnecessary stride
sorting and guard creation.to avoid unnecessary recomputation of these properties.

* Modify your code to make precomputed properties *lazy*. This ensures that
guards on unbacked symbolic integers are only applied when necessary,
reducing computational overhead.

* Use the `constrain_range` API to define bounds for integer tensor sizes.
This helps maintain tensor sizes within specified limits, enhancing control
and predictability.

## How to use unbacked
To use unbacked APIs, replace `mark_dynamic` with `mark_unbacked` and
`TORCH_COMPILE_DYNAMIC_SOURCES` with `TORCH_COMPILE_UNBACKED_SOURCES`.
This tells the compiler to treat an input as unbacked.

```{seealso}
* {ref}`dynamic_shapes_overview`
* {ref}`torch.export`
* {ref}`what_is_a_specialization`
```
