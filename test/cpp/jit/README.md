# JIT CPP Tests

## How to add a new test

First, create a new test file. Test files should have be placed in this
directory, with a name that starts with `test_`, like `test_foo.cpp`.

Here is an example test file you can copy-paste.
```cpp
#include <test/cpp/jit/test_base.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCaseOne() {
    // ...
}

void testCaseTwo() {
    // ...
}
}
}
```

Then, register your test in `tests.h`:
```cpp
// Add to TH_FORALL_TESTS_CUDA instead for CUDA-requiring tests
#define TH_FORALL_TESTS(_)             \
  _(ADFormulas)                        \
  _(Attributes)                        \
  ...
  _(CaseOne)  // note that the `test` prefix is omitted.
  _(CaseTwo)
```
