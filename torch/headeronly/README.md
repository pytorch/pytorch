## torch/headeronly

The inlined C++ headers in the `torch::headeronly` namespace living this subdirectory are completely decoupled from LibTorch. These APIs are also globally listed in [torch/header_only_apis.txt](https://github.com/pytorch/pytorch/blob/main/torch/header_only_apis.txt).

There are two types of LibTorch independent header-only headers:
1. OG header-only. Originally header-only APIs, such as `ScalarType`, `Half`, `BFloat16`, have always been implemented in headers only. For them to move into torch/headeronly only required a code migration, a copy-pasta, if you will.
2. Made to be header-only. There are also APIs that were NOT header-only that we made to be header-only. One example of such an API is `STD_TORCH_CHECK`, which was derived from `TORCH_CHECK`. `STD_TORCH_CHECK` calls into `std::runtime_error` instead of relying on `c10::Error`, which relies on libtorch.so. As a result, `STD_TORCH_CHECK` does not have the full `TORCH_CHECK` functionality that displays a fanciful traceback when the check is not met. We intentionally maintain the design that functions that do different things should be explicitly named differently.
