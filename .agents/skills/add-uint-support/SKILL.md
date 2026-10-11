---
name: add-uint-support
description: Add unsigned integer (uint) type support to PyTorch operators by updating AT_DISPATCH macros. Use when adding support for uint16, uint32, uint64 types to operators, kernels, or when user mentions enabling unsigned types, barebones unsigned types, or uint support.
---

# Add Unsigned Integer (uint) Support to Operators

This skill helps add support for unsigned integer types (uint16, uint32, uint64) to PyTorch operators by updating their AT_DISPATCH macros.

## When to use this skill

Use this skill when:
- Adding uint16, uint32, or uint64 support to an operator
- User mentions "unsigned types", "uint support", "barebones unsigned types"
- Enabling support for kUInt16, kUInt32, kUInt64 in kernels
- Working with operator implementations that need expanded type coverage

## Quick reference

**Add unsigned types to existing dispatch:**
```cpp
// Before
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES));

// After (method 1: add unsigned types explicitly)
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));

// After (method 2: use V2 integral types if AT_INTEGRAL_TYPES present)
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_INTEGRAL_TYPES_V2), AT_EXPAND(AT_FLOATING_TYPES));
```

## Type group reference

**Unsigned type groups:**
- `AT_BAREBONES_UNSIGNED_TYPES`: kUInt16, kUInt32, kUInt64
- `AT_INTEGRAL_TYPES_V2`: AT_INTEGRAL_TYPES + AT_BAREBONES_UNSIGNED_TYPES

**Relationship:**
```cpp
AT_INTEGRAL_TYPES          // kByte, kChar, kInt, kLong, kShort
AT_BAREBONES_UNSIGNED_TYPES  // kUInt16, kUInt32, kUInt64
AT_INTEGRAL_TYPES_V2       // INTEGRAL_TYPES + BAREBONES_UNSIGNED_TYPES
```

## Instructions

### Step 1: Determine if conversion to V2 is needed

Check if the file uses AT_DISPATCH_V2:

**If using old AT_DISPATCH:**
- First convert to AT_DISPATCH_V2 using the at-dispatch-v2 skill
- Then proceed with adding uint support

**If already using AT_DISPATCH_V2:**
- Proceed directly to Step 2

### Step 2: Analyze the current dispatch macro

Identify what type groups are currently in use:

```cpp
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  // body
}), AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Current type coverage
```

Common patterns:
- `AT_EXPAND(AT_ALL_TYPES)` → includes AT_INTEGRAL_TYPES + AT_FLOATING_TYPES
- `AT_EXPAND(AT_INTEGRAL_TYPES)` → signed integers only
- `AT_EXPAND(AT_FLOATING_TYPES)` → floating point types

### Step 3: Choose the uint addition method

Two approaches:

**Method 1: Add AT_BAREBONES_UNSIGNED_TYPES explicitly**
- Use when: You want to be explicit about adding uint support
- Add `AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)` to the type list

**Method 2: Substitute AT_INTEGRAL_TYPES with AT_INTEGRAL_TYPES_V2**
- Use when: The dispatch already uses `AT_EXPAND(AT_INTEGRAL_TYPES)`
- More concise: replaces one type group with its superset
- Only applicable if AT_INTEGRAL_TYPES is present

### Step 4: Apply the transformation

**Method 1 example:**
```cpp
// Before
AT_DISPATCH_V2(
    dtype,
    "min_values_cuda",
    AT_WRAP([&]() {
      kernel_impl<scalar_t>(iter);
    }),
    AT_EXPAND(AT_ALL_TYPES),
    kBFloat16, kHalf, kBool
);

// After (add unsigned types)
AT_DISPATCH_V2(
    dtype,
    "min_values_cuda",
    AT_WRAP([&]() {
      kernel_impl<scalar_t>(iter);
    }),
    AT_EXPAND(AT_ALL_TYPES),
    AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
    kBFloat16, kHalf, kBool
);
```

**Method 2 example:**
```cpp
// Before
AT_DISPATCH_V2(
    dtype,
    "integral_op",
    AT_WRAP([&]() {
      kernel<scalar_t>();
    }),
    AT_EXPAND(AT_INTEGRAL_TYPES)
);

// After (substitute with V2)
AT_DISPATCH_V2(
    dtype,
    "integral_op",
    AT_WRAP([&]() {
      kernel<scalar_t>();
    }),
    AT_EXPAND(AT_INTEGRAL_TYPES_V2)
);
```

### Step 5: Handle AT_ALL_TYPES vs individual type groups

If the dispatch uses `AT_EXPAND(AT_ALL_TYPES)`:
- `AT_ALL_TYPES` = `AT_INTEGRAL_TYPES` + `AT_FLOATING_TYPES`
- To add uint: add `AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)` to the list

If the dispatch separately lists INTEGRAL and FLOATING:
```cpp
// Before
AT_EXPAND(AT_INTEGRAL_TYPES), AT_EXPAND(AT_FLOATING_TYPES)

// After (Method 2 preferred)
AT_EXPAND(AT_INTEGRAL_TYPES_V2), AT_EXPAND(AT_FLOATING_TYPES)
```

### Step 6: Verify all dispatch sites

Check the file for ALL dispatch macros that need uint support:
- Some operators have multiple dispatch sites (CPU, CUDA, different functions)
- Apply the transformation consistently across all sites
- Ensure each gets the same type coverage updates

### Step 7: Validate the changes

Check that:
- [ ] AT_DISPATCH_V2 format is used (not old AT_DISPATCH)
- [ ] Unsigned types are added via one of the two methods
- [ ] All relevant dispatch sites in the file are updated
- [ ] Type groups use `AT_EXPAND()`
- [ ] Arguments are properly formatted and comma-separated

## Common patterns

### Pattern 1: AT_ALL_TYPES + extras

```cpp
// Before
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);

// After
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kHalf, kBFloat16);
```

### Pattern 2: Separate INTEGRAL + FLOATING

```cpp
// Before
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_INTEGRAL_TYPES), AT_EXPAND(AT_FLOATING_TYPES));

// After
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_INTEGRAL_TYPES_V2), AT_EXPAND(AT_FLOATING_TYPES));
```

### Pattern 3: Old dispatch needs conversion first

```cpp
// Before (needs v2 conversion first)
AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "op", [&]() {
  kernel<scalar_t>();
});

// After v2 conversion
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);

// After adding uint support
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kHalf, kBFloat16);
```

## Multiple dispatch sites example

For a file with multiple functions:

```cpp
void min_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_V2(iter.dtype(), "min_values_cuda", AT_WRAP([&]() {
    impl<scalar_t>(iter);
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf);
  //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                           Added uint support
}

void min_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_V2(iter.input_dtype(), "min_cuda", AT_WRAP([&]() {
    gpu_reduce_kernel<scalar_t>(iter);
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf);
  //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                           Added uint support here too
}
```

## Decision tree

Use this decision tree to determine the approach:

```
Is the file using AT_DISPATCH_V2?
├─ No → Use at-dispatch-v2 skill first, then continue
└─ Yes
   └─ Does it use AT_EXPAND(AT_INTEGRAL_TYPES)?
      ├─ Yes → Replace with AT_EXPAND(AT_INTEGRAL_TYPES_V2)
      └─ No → Add AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES) to type list
```

## Edge cases

### Case 1: Dispatch with only floating types

If the operator only supports floating point types, don't add uint support:

```cpp
// Leave as-is - floating point only operator
AT_DISPATCH_V2(dtype, "float_op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_FLOATING_TYPES), kHalf);
```

### Case 2: Complex types present

Unsigned types work alongside complex types:

```cpp
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&]() {
  kernel<scalar_t>();
}), AT_EXPAND(AT_ALL_TYPES),
    AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
    AT_EXPAND(AT_COMPLEX_TYPES),
    kHalf, kBFloat16);
```

### Case 3: Already has uint support

Check if uint types are already present:
- If `AT_INTEGRAL_TYPES_V2` is used → already has uint support
- If `AT_BAREBONES_UNSIGNED_TYPES` is already in list → already has uint support
- Skip the file if uint support is already present

## Workflow

When asked to add uint support:

1. Read the target file
2. Check if using AT_DISPATCH_V2:
   - If not → use at-dispatch-v2 skill first
3. Identify all dispatch macro sites
4. For each dispatch:
   - Analyze current type groups
   - Choose method (add BAREBONES_UNSIGNED or upgrade to V2)
   - Apply transformation with Edit tool
5. Show the user the changes
6. Explain what was modified

## Important notes

- Always check if v2 conversion is needed first
- Apply changes consistently across all dispatch sites in the file
- Method 2 (AT_INTEGRAL_TYPES_V2) is cleaner when applicable
- Method 1 (explicit AT_BAREBONES_UNSIGNED_TYPES) is more explicit
- Unsigned types are: kUInt16, kUInt32, kUInt64 (not kByte which is uint8)
- Some operators may not semantically support unsigned types - use judgment

## Testing

After adding uint support, the operator should accept uint16, uint32, and uint64 tensors. The user is responsible for functional testing.
