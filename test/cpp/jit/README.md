# JIT C++ Tests

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

We glob all the test files together in `CMakeLists.txt` so that you don't
have to edit it every time you add a test. Unfortunately, this means that in
order to get the build to pick up your new test file, you need to re-run
cmake:
```
python setup.py build --cmake
```

## Why do we have two different test runners?
We have two different ways of running our cpp tests:
1. With `gtest`, from a standalone binary.
2. With Python, from `TestJit.test_cpp` and `TestJit.test_cpp_cuda` (in
   `test/test_jit.py`)

We want both because we need to test things from a pure-C++ environment and
with all our various Python patch-points enabled.

## How do I run the tests?
The following commands assume you are in PyTorch root.

1. With `gtest`:
   ```bash
   # (re)build the test binary
   ninja build/bin/test_jit
   # run
   build/bin/test_jit --gtest_filter='glob_style_filter*'
   ```
2. With Python:
   ```
   python test/test_jit.py TestJit.test_cpp TestJit.test_cpp_cuda
   ```
