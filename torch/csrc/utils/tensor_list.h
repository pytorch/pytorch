#pragma once

#include <ATen/ATen.h>
#include <Python.h>

namespace torch {

PyObject* THPUtils_tensorToList(const at::Tensor& tensor);

}
