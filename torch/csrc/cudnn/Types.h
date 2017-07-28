#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <string>
#include <cudnn.h>
#include "../Types.h"
#include <ATen/Tensor.h>

namespace torch { namespace cudnn {

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);
cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);
void _THVoidTensor_assertContiguous(THVoidTensor *tensor, const std::string& name);

#define THVoidTensor_assertContiguous(tensor) \
_THVoidTensor_assertContiguous(tensor, #tensor " tensor")

}}  // namespace torch::cudnn

#endif
