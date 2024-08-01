#pragma once

#include <torch/csrc/python_headers.h>

namespace at {
class Tensor;
}

namespace torch::utils {

PyObject* tensor_to_list(const at::Tensor& tensor);

}
