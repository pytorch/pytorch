// ${generated_comment}

#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;

namespace torch { namespace autograd {

namespace {

inline PyObject* wrap(Tensor tensor) {
  return THPVariable_Wrap(Variable(std::move(tensor)));
}

inline PyObject* wrap(std::tuple<Tensor, Tensor> tensors) {
  auto tuple = THPObjectPtr{PyTuple_New(2)};
  if (!tuple) return NULL;
  PyTuple_SET_ITEM(tuple.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(tuple.get(), 1, wrap(std::move(std::get<1>(tensors))));
  return tuple.release();
}

inline PyObject* wrap(std::tuple<Tensor, Tensor, Tensor> tensors) {
  auto tuple = THPObjectPtr{PyTuple_New(3)};
  if (!tuple) return NULL;
  PyTuple_SET_ITEM(tuple.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(tuple.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(tuple.get(), 2, wrap(std::move(std::get<2>(tensors))));
  return tuple.release();
}

inline PyObject* wrap(bool value) {
  if (value) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

inline PyObject* wrap(int64_t value) {
  return THPUtils_packInt64(value);
}

inline PyObject* wrap(Scalar scalar) {
  return wrap(scalar.toTensor());
}

} // anonymous namespace

${py_methods}

PyMethodDef variable_methods[] = {
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
