# JIT C++ Tests

## Adding a new test
First, create a new test file. Test files should have be placed in this
directory, with a name that starts with `test_`, like `test_foo.cpp`.

In general a single test suite

Add your test file to the `JIT_TEST_SRCS` list in `test/cpp/jit/CMakeLists.txt`.

A test file may look like:
```cpp
#include <gtest/gtest.h>

using namespace ::torch::jit

TEST(FooTest, BarBaz) {
   // ...
}

// Append '_CUDA' to the test case name will automatically filter it out if CUDA
// is not compiled.
TEST(FooTest, NeedsAGpu_CUDA) {
   // ...
}

// Similarly, if only one GPU is detected, tests with `_MultiCUDA` at the end
// will not be run.
TEST(FooTest, NeedsMultipleGpus_MultiCUDA) {
   // ...
}
```

## Building and running the tests
The following commands assume you are in PyTorch root.

```bash
# ... Build PyTorch from source, e.g.
python setup.py develop
# (re)build just the binary
ninja -C build bin/test_jit
# run tests
build/bin/test_jit --gtest_filter='glob_style_filter*'
```
