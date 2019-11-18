
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/nested_tensor/python_nested_tensor.h>
namespace torch {
namespace nested_tensor {

PyObject *_ListNestedTensorVariable___repr__(PyObject *self_);

PyObject *_ListNestedTensorVariable___str__(PyObject *self_);

PyObject *_ListNestedTensorVariable_detach(PyObject *self_);

PyObject *_ListNestedTensorVariable_dim(PyObject *self_);

PyObject *_ListNestedTensorVariable_element_size(PyObject *self_);

PyObject *_ListNestedTensorVariable_grad(PyObject *self_);

PyObject *_ListNestedTensorVariable_is_contiguous(PyObject *self_);

PyObject *_ListNestedTensorVariable_is_pinned(PyObject *self_);

PyObject *_ListNestedTensorVariable_nested_dim(PyObject *self_);

PyObject *_ListNestedTensorVariable_numel(PyObject *self_);

PyObject *_ListNestedTensorVariable_pin_memory(PyObject *self_);

PyObject *_ListNestedTensorVariable_requires_grad(PyObject *self_);

PyObject *_ListNestedTensorVariable_to_tensor(PyObject *self_);

}
}
