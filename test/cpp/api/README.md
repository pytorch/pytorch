# C++ API Tests

In this folder live the tests for PyTorch's C++ API (formerly known as
autogradpp). They use the [Catch2](https://github.com/catchorg/Catch2) test
framework.

## CUDA Tests

The way we handle CUDA tests is by separating them into a separate `TEST_CASE`
(e.g. we have `optim` and `optim_cuda` test cases in `optim.cpp`), and giving
them the `[cuda]` tag. Then, inside `main.cpp` we detect at runtime whether
CUDA is available. If not, we disable these CUDA tests by appending `~[cuda]`
to the test specifications. The `~` disables the tag.

One annoying aspect is that Catch only allows filtering on test cases and not
sections. Ideally, one could have a section like `LSTM` inside the `RNN` test
case, and give this section a `[cuda]` tag to only run it when CUDA is
available. Instead, we have to create a whole separate `RNN_cuda` test case and
put all these CUDA sections in there.

## Integration Tests

Integration tests use the MNIST dataset. You must download it by running the
following command from the PyTorch root folder:

```shell
$ python tools/download_mnist.py -d test/cpp/api/mnist
```
