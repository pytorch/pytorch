#pragma once

#include <Python.h>
#include "convolution.h"
#include "torch/csrc/autograd/python_cpp_function_attr.h"


namespace torch { namespace autograd {

namespace attributes {

template<class T>
PyObject* conv_stride(THPCppFunction* self, PyObject* hook)
{
  auto& stride = std::static_pointer_cast<T>(self->cdata)->stride;
  auto num_elems = stride.size();
  THPObjectPtr py_stride = PyTuple_New(num_elems);
  if (!py_stride) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_stride.get(), i, PyLong_FromLong(stride[i]));
  }
  return py_stride.release();
}

template<class T>
PyObject* conv_padding(THPCppFunction* self, PyObject* hook)
{
  auto& padding = std::static_pointer_cast<T>(self->cdata)->padding;
  auto num_elems = padding.size();
  THPObjectPtr py_padding = PyTuple_New(num_elems);
  if (!py_padding) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_padding.get(), i, PyLong_FromLong(padding[i]));
  }
  return py_padding.release();
}

template<class T>
PyObject* conv_dilation(THPCppFunction* self, PyObject* hook)
{
  auto& dilation = std::static_pointer_cast<T>(self->cdata)->dilation;
  auto num_elems = dilation.size();
  THPObjectPtr py_dilation = PyTuple_New(num_elems);
  if (!py_dilation) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_dilation.get(), i, PyLong_FromLong(dilation[i]));
  }
  return py_dilation.release();
}

template<class T>
PyObject* conv_transposed(THPCppFunction* self, PyObject* hook)
{
  bool transposed = std::static_pointer_cast<T>(self->cdata)->transposed;
  if (transposed) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

template<class T>
PyObject* conv_output_padding(THPCppFunction* self, PyObject* hook)
{
  auto& output_padding = std::static_pointer_cast<T>(self->cdata)->output_padding;
  auto num_elems = output_padding.size();
  THPObjectPtr py_output_padding = PyTuple_New(num_elems);
  if (!py_output_padding) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_output_padding.get(), i, PyLong_FromLong(output_padding[i]));
  }
  return py_output_padding.release();
}

template<class T>
PyObject* conv_groups(THPCppFunction* self, PyObject* hook)
{
  int groups = std::static_pointer_cast<T>(self->cdata)->groups;
  return PyLong_FromLong(groups);
}

static struct PyGetSetDef conv_forward_properties[] = {
  {(char*)"stride", (getter)conv_stride<ConvForward>, NULL, NULL, NULL},
  {(char*)"padding", (getter)conv_padding<ConvForward>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)conv_dilation<ConvForward>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)conv_transposed<ConvForward>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)conv_output_padding<ConvForward>, NULL, NULL, NULL},
  {(char*)"groups", (getter)conv_groups<ConvForward>, NULL, NULL, NULL},
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef conv_backward_properties[] = {
  {(char*)"stride", (getter)conv_stride<ConvBackward>, NULL, NULL, NULL},
  {(char*)"padding", (getter)conv_padding<ConvBackward>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)conv_dilation<ConvBackward>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)conv_transposed<ConvBackward>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)conv_output_padding<ConvBackward>, NULL, NULL, NULL},
  {(char*)"groups", (getter)conv_groups<ConvBackward>, NULL, NULL, NULL},
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

}

}}
