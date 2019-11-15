
#include <torch/csrc/nested_tensor/generated/dispatch.h>

PyObject *_ListNestedTensorVariable___repr__(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.__repr__());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable___str__(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.__str__());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_detach(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.detach());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_device(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.device());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_dim(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.dim());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_dtype(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.dtype());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_element_size(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.element_size());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_grad(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.grad());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_is_contiguous(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.is_contiguous());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_is_pinned(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.is_pinned());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_layout(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.layout());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_numel(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.numel());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_pin_memory(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.pin_memory());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_requires_grad(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.requires_grad());
  END_HANDLE_TH_ERRORS
}

PyObject *_ListNestedTensorVariable_to_tensor(PyObject *self_) {
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return wrap(self.to_tensor());
  END_HANDLE_TH_ERRORS
}
