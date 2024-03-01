# C++ Frontend Tests

In this folder live the tests for PyTorch's C++ Frontend. They use the
[GoogleTest](https://github.com/google/googletest) test framework.

## CUDA Tests

To make a test runnable only on platforms with CUDA, you should suffix your
test with `_CUDA`, e.g.

```cpp
TEST(MyTestSuite, MyTestCase_CUDA) { }
```

To make it runnable only on platforms with at least two CUDA machines, suffix
it with `_MultiCUDA` instead of `_CUDA`, e.g.

```cpp
TEST(MyTestSuite, MyTestCase_MultiCUDA) { }
```

There is logic in `main.cpp` that detects the availability and number of CUDA
devices and supplies the appropriate negative filters to GoogleTest.

## Integration Tests

Integration tests use the MNIST dataset. You must download it by running the
following command from the PyTorch root folder:

```sh
$ python tools/download_mnist.py -d test/cpp/api/mnist
```

The required paths will be referenced as `test/cpp/api/mnist/...` in the test
code, so you *must* run the integration tests from the PyTorch root folder.
