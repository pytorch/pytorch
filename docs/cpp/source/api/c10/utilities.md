---
myst:
  html_meta:
    description: C10 utility classes in PyTorch C++ — Flags, QEngine, and Reduction enumerations.
    keywords: PyTorch, C++, c10, utilities, QEngine, Reduction
---

# Utilities

C10 provides utility classes for memory management and other common patterns.

## MaybeOwned<Tensor>

`MaybeOwned<Tensor>` is a C++ smart pointer class that dynamically
encodes whether a Tensor is *owned* or *borrowed*. It is used in
certain performance-sensitive situations to avoid unnecessarily
incrementing a Tensor's reference count (at a small cost in
overhead from the extra indirection).

```{warning}

 MaybeOwned must be used with **extreme** care. Claims of (non-)ownership
 are not statically checked, and mistakes can cause reference undercounting
 and use-after-free crashes.

 Due to this lack of safety net, we discourage the use of MaybeOwned
 outside code paths that are known to be highly performance sensitive.
 However, if you encounter pre-existing uses of MaybeOwned in code that
 you want to modify, it's critical to understand how to use it correctly.
```

**Use Case:**

The primary use case for `MaybeOwned<Tensor>` is a function or method that
dynamically chooses between returning one of its arguments (typically
from a passthrough or "no-op" code path) and returning a freshly constructed
Tensor. Such a function would return a `MaybeOwned<Tensor>` in both cases:
the former in a "borrowed" state via `MaybeOwned<Tensor>::borrowed()`,
and the latter in an "owned" state via `MaybeOwned<Tensor>::owned()`.

**Example - expect_contiguous:**

The canonical example is `Tensor`'s `expect_contiguous` method, which shortcuts
and returns a borrowed self-reference when already contiguous:

```cpp
inline c10::MaybeOwned<Tensor> Tensor::expect_contiguous(
    MemoryFormat memory_format) const & {
  if (is_contiguous(memory_format)) {
    return c10::MaybeOwned<Tensor>::borrowed(*this);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        __dispatch_contiguous(memory_format));
  }
}
```

Using the vocabulary of lifetimes, the essential safety requirement for borrowing
is that a borrowed Tensor must outlive any borrowing references to it. In the example
above, we can safely borrow `*this`, but the Tensor returned by
`__dispatch_contiguous()` is freshly created, and borrowing a reference would
effectively leave it ownerless.

**Rules of Thumb:**

- When in doubt, don't use `MaybeOwned<Tensor>` at all - in particular, prefer
  avoiding using it in code that doesn't use it already. New usage should only be
  introduced when critical (and demonstrable) performance gains result.

- When modifying or calling code that already uses `MaybeOwned<Tensor>`, remember
  that it's always safe to produce a `MaybeOwned<Tensor>` from a Tensor in hand
  via a call to `MaybeOwned<Tensor>::owned()`. This may result in an unnecessary
  reference count, but never in misbehavior - so it's always the safer bet, unless
  the lifetime of the Tensor you're looking to wrap is crystal clear.

More details and implementation code can be found at
[MaybeOwned.h](https://github.com/pytorch/pytorch/blob/main/c10/util/MaybeOwned.h) and
[TensorBody.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/templates/TensorBody.h).

## Error Handling and Assertions

PyTorch provides macros for error checking and assertions that produce
informative error messages with source location. These are defined in
`c10/util/Exception.h`.

### TORCH_CHECK

The primary macro for validating user input and runtime conditions. On failure,
raises `c10::Error` (which becomes `RuntimeError` in Python).

```cpp
#include <c10/util/Exception.h>

// Basic check
TORCH_CHECK(tensor.dim() == 2, "Expected 2D tensor, got ", tensor.dim(), "D");

// Without message (default message generated)
TORCH_CHECK(x >= 0);
```

Typed variants raise specific Python exception types:

- `TORCH_CHECK_INDEX(cond, ...)` — raises `IndexError`
- `TORCH_CHECK_VALUE(cond, ...)` — raises `ValueError`
- `TORCH_CHECK_TYPE(cond, ...)` — raises `TypeError`
- `TORCH_CHECK_LINALG(cond, ...)` — raises `LinAlgError`
- `TORCH_CHECK_NOT_IMPLEMENTED(cond, ...)` — raises `NotImplementedError`

### TORCH_INTERNAL_ASSERT

For internal invariants that should always hold (i.e., failures indicate a bug
in PyTorch, not user error). Produces a message asking users to report the bug.

```cpp
TORCH_INTERNAL_ASSERT(googol > 0);
TORCH_INTERNAL_ASSERT(googol > 0, "googol was ", googol);
```

```{note}

Use `TORCH_CHECK` for conditions that can fail due to user input.
Use `TORCH_INTERNAL_ASSERT` only for conditions that indicate a PyTorch bug.
`TORCH_INTERNAL_ASSERT_DEBUG_ONLY` is the debug-build-only variant for
hot paths.
```

### TORCH_WARN

Issues a warning (not an error) to the user.

```cpp
TORCH_WARN("This operation is slow for sparse tensors");
TORCH_WARN_ONCE("This warning appears only once");
```

### c10::Error

The base exception class for PyTorch C++ errors. Provides source location
and optional backtrace.

```cpp
try {
    auto result = some_operation();
} catch (const c10::Error& e) {
    std::cerr << e.what() << std::endl;
    // Or without backtrace:
    std::cerr << e.what_without_backtrace() << std::endl;
}
```

Specialized subclasses: `c10::IndexError`, `c10::ValueError`,
`c10::TypeError`, `c10::NotImplementedError`, `c10::LinAlgError`,
`c10::OutOfMemoryError`.
