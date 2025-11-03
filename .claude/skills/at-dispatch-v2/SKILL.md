---
name: at-dispatch-v2
description: Convert PyTorch AT_DISPATCH macros to AT_DISPATCH_V2 format in ATen C++ code. Use when porting AT_DISPATCH_ALL_TYPES_AND*, AT_DISPATCH_FLOATING_TYPES*, or other dispatch macros to the new v2 API. For ATen kernel files, CUDA kernels, and native operator implementations.
---

# AT_DISPATCH to AT_DISPATCH_V2 Converter

This skill helps convert PyTorch's legacy AT_DISPATCH macros to the new AT_DISPATCH_V2 format, as defined in `aten/src/ATen/Dispatch_v2.h`.

## When to use this skill

Use this skill when:
- Converting AT_DISPATCH_* macros to AT_DISPATCH_V2
- Porting ATen kernels to use the new dispatch API
- Working with files in `aten/src/ATen/native/` that use dispatch macros
- User mentions "AT_DISPATCH", "dispatch v2", "Dispatch_v2.h", or macro conversion

## Quick reference

**Old format:**
```cpp
AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, dtype, "kernel_name", [&]() {
  // lambda body
});
```

**New format:**
```cpp
AT_DISPATCH_V2(dtype, "kernel_name", AT_WRAP([&]() {
  // lambda body
}), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kHalf, kBool);
```

## Key transformations

1. **Reorder arguments**: `scalar_type` and `name` come first, then lambda, then types
2. **Wrap the lambda**: Use `AT_WRAP(lambda)` to handle internal commas
3. **Expand type groups**: Use `AT_EXPAND(AT_ALL_TYPES)` instead of implicit expansion
4. **List individual types**: Add extra types (kHalf, kBFloat16, etc.) after expanded groups
5. **Add include**: `#include <ATen/Dispatch_v2.h>` near other Dispatch includes

## Instructions

### Step 1: Add the Dispatch_v2.h include

Add the v2 header near the existing `#include <ATen/Dispatch.h>`:

```cpp
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
```

Keep the old Dispatch.h include for now (other code may still need it).

### Step 2: Identify the old dispatch pattern

Common patterns to convert:

- `AT_DISPATCH_ALL_TYPES_AND{2,3,4}(type1, type2, ..., scalar_type, name, lambda)`
- `AT_DISPATCH_FLOATING_TYPES_AND{2,3}(type1, type2, ..., scalar_type, name, lambda)`
- `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND{2,3}(type1, ..., scalar_type, name, lambda)`
- `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND{2,3}(type1, ..., scalar_type, name, lambda)`

### Step 3: Map the old macro to type groups

Identify which type group macro corresponds to the base types:

| Old macro base | AT_DISPATCH_V2 type group |
|----------------|---------------------------|
| `ALL_TYPES` | `AT_EXPAND(AT_ALL_TYPES)` |
| `FLOATING_TYPES` | `AT_EXPAND(AT_FLOATING_TYPES)` |
| `INTEGRAL_TYPES` | `AT_EXPAND(AT_INTEGRAL_TYPES)` |
| `COMPLEX_TYPES` | `AT_EXPAND(AT_COMPLEX_TYPES)` |
| `ALL_TYPES_AND_COMPLEX` | `AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX)` |

For combined patterns, use multiple `AT_EXPAND()` entries:
```cpp
// Old: AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(...)
// New: AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_COMPLEX_TYPES), type1, type2
```

### Step 4: Extract the individual types

From `AT_DISPATCH_*_AND2(type1, type2, ...)` or `AT_DISPATCH_*_AND3(type1, type2, type3, ...)`, extract the individual types (type1, type2, etc.).

These become the trailing arguments after the type group:
```cpp
AT_DISPATCH_V2(..., AT_EXPAND(AT_ALL_TYPES), kBFloat16, kHalf, kBool)
                                             ^^^^^^^^^^^^^^^^^^^^^^^^
                                             Individual types from AND3
```

### Step 5: Transform to AT_DISPATCH_V2

Apply the transformation:

**Pattern:**
```cpp
AT_DISPATCH_V2(
  scalar_type,           // 1st: The dtype expression
  "name",                // 2nd: The debug string
  AT_WRAP(lambda),       // 3rd: The lambda wrapped in AT_WRAP
  type_groups,           // 4th+: Type groups with AT_EXPAND()
  individual_types       // Last: Individual types
)
```

**Example transformation:**
```cpp
// BEFORE
AT_DISPATCH_ALL_TYPES_AND3(
    kBFloat16, kHalf, kBool,
    iter.dtype(),
    "min_values_cuda",
    [&]() {
      min_values_kernel_cuda_impl<scalar_t>(iter);
    }
);

// AFTER
AT_DISPATCH_V2(
    iter.dtype(),
    "min_values_cuda",
    AT_WRAP([&]() {
      min_values_kernel_cuda_impl<scalar_t>(iter);
    }),
    AT_EXPAND(AT_ALL_TYPES),
    kBFloat16, kHalf, kBool
);
```

### Step 6: Handle multi-line lambdas

For lambdas with internal commas or complex expressions, AT_WRAP is essential:

```cpp
AT_DISPATCH_V2(
    dtype,
    "complex_kernel",
    AT_WRAP([&]() {
      gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        MinOps<scalar_t>{},
        thrust::pair<scalar_t, int64_t>(upper_bound(), 0)  // Commas inside!
      );
    }),
    AT_EXPAND(AT_ALL_TYPES)
);
```

### Step 7: Verify the conversion

Check that:
- [ ] `AT_WRAP()` wraps the entire lambda
- [ ] Type groups use `AT_EXPAND()`
- [ ] Individual types don't have `AT_EXPAND()` (just `kBFloat16`, not `AT_EXPAND(kBFloat16)`)
- [ ] Argument order is: scalar_type, name, lambda, types
- [ ] Include added: `#include <ATen/Dispatch_v2.h>`

## Type group reference

Available type group macros (use with `AT_EXPAND()`):

```cpp
AT_INTEGRAL_TYPES      // kByte, kChar, kInt, kLong, kShort
AT_FLOATING_TYPES      // kDouble, kFloat
AT_COMPLEX_TYPES       // kComplexDouble, kComplexFloat
AT_QINT_TYPES         // kQInt8, kQUInt8, kQInt32
AT_ALL_TYPES          // INTEGRAL_TYPES + FLOATING_TYPES
AT_ALL_TYPES_AND_COMPLEX  // ALL_TYPES + COMPLEX_TYPES
AT_INTEGRAL_TYPES_V2  // INTEGRAL_TYPES + unsigned types
AT_BAREBONES_UNSIGNED_TYPES  // kUInt16, kUInt32, kUInt64
AT_FLOAT8_TYPES       // Float8 variants
```

## Common patterns

### Pattern: AT_DISPATCH_ALL_TYPES_AND2

```cpp
// Before
AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "op", [&]() {
  kernel<scalar_t>(data);
});

// After
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>(data);
}), AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);
```

### Pattern: AT_DISPATCH_FLOATING_TYPES_AND3

```cpp
// Before
AT_DISPATCH_FLOATING_TYPES_AND3(kHalf, kBFloat16, kFloat8_e4m3fn,
    tensor.scalar_type(), "float_op", [&] {
  process<scalar_t>(tensor);
});

// After
AT_DISPATCH_V2(tensor.scalar_type(), "float_op", AT_WRAP([&] {
  process<scalar_t>(tensor);
}), AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16, kFloat8_e4m3fn);
```

### Pattern: AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2

```cpp
// Before
AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kComplexHalf, kHalf,
    self.scalar_type(),
    "complex_op",
    [&] {
      result = compute<scalar_t>(self);
    }
);

// After
AT_DISPATCH_V2(
    self.scalar_type(),
    "complex_op",
    AT_WRAP([&] {
      result = compute<scalar_t>(self);
    }),
    AT_EXPAND(AT_ALL_TYPES),
    AT_EXPAND(AT_COMPLEX_TYPES),
    kComplexHalf,
    kHalf
);
```

## Edge cases

### Case 1: No extra types (rare)

```cpp
// Before
AT_DISPATCH_ALL_TYPES(dtype, "op", [&]() { kernel<scalar_t>(); });

// After
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES));
```

### Case 2: Many individual types (AND4, AND5, etc.)

```cpp
// Before
AT_DISPATCH_FLOATING_TYPES_AND4(kHalf, kBFloat16, kFloat8_e4m3fn, kFloat8_e5m2,
    dtype, "float8_op", [&]() { kernel<scalar_t>(); });

// After
AT_DISPATCH_V2(dtype, "float8_op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16, kFloat8_e4m3fn, kFloat8_e5m2);
```

### Case 3: Lambda with no captures

```cpp
// Before
AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, dtype, "op", []() {
  static_kernel<scalar_t>();
});

// After
AT_DISPATCH_V2(dtype, "op", AT_WRAP([]() {
  static_kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), kHalf, kBool);
```

## Benefits of AT_DISPATCH_V2

1. **No arity in macro name**: Don't need different macros for AND2, AND3, AND4
2. **Composable type sets**: Mix and match type groups with `AT_EXPAND()`
3. **Extensible**: Easy to add more types without hitting macro limits
4. **Clearer**: Type groups are explicit, not implicit in macro name

## Important notes

- Keep `#include <ATen/Dispatch.h>` - other code may need it
- The `AT_WRAP()` is mandatory - prevents comma parsing issues in the lambda
- Type groups need `AT_EXPAND()`, individual types don't
- The v2 API is in `aten/src/ATen/Dispatch_v2.h` - refer to it for full docs
- See the header file for the Python script to regenerate the macro implementation

## Workflow

When asked to convert AT_DISPATCH macros:

1. Read the file to identify all AT_DISPATCH uses
2. Add `#include <ATen/Dispatch_v2.h>` if not present
3. For each dispatch macro:
   - Identify the pattern and extract components
   - Map the base type group
   - Extract individual types
   - Construct the AT_DISPATCH_V2 call
   - Apply with Edit tool
4. Show the user the complete converted file
5. Explain what was changed

Do NOT compile or test the code - focus on accurate conversion only.
