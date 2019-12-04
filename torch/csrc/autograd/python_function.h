#pragma once

#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <c10/util/Optional.h>
#include <c10/core/DeviceGuard.h>

#include <vector>
#include <utility>
#include <memory>

namespace torch { namespace jit { struct Graph; }}
namespace torch { namespace autograd {

// A Function which is implemented by a Python object (i.e., a THPFunction).
// Calls to 'apply' are forwarded to the Python method implementation.
struct PyNode : public Node {
  PyNode(THPObjectPtr obj) : obj(obj.release()) {}

  variable_list apply(variable_list&& inputs) override;
  variable_list legacy_apply(const variable_list& inputs);

  void release_variables() override;
  std::string name() const override;
  bool is_traceable() override;

  // THPFunction this Function is wrapping.  Owning!
  PyObject* obj;

  ~PyNode() {
    // Can't use THPObjectPtr as a field in this class; destructor won't take
    // out GIL!  When I forgot to do this by hand
    // TestAutograd.test_inplace_view_python called me out about it.
    pybind11::gil_scoped_acquire g;
    Py_DECREF(obj);
  }
};

/**
 * Cast an object into a tuple, if it is not a tuple already. Returns true
 * if the original object was not a tuple.
 */
inline bool ensure_tuple(THPObjectPtr& obj) {
  if (PyTuple_Check(obj.get()))
    return false;

  PyObject *tuple = PyTuple_New(1);
  if (!tuple) throw python_error();
  PyTuple_SET_ITEM(tuple, 0, obj.release());
  obj = tuple;
  return true;
}

}} // namespace torch::autograd

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;

    // Python tuple of tensors whose variables we should save.  Set
    // by Python with 'save_for_backward'.  If nullptr, no tensors were
    // saved.
    PyObject *to_save;
    // Python tuple of tensors which are not differentiable.  Set by
    // Python with 'mark_non_differentiable'.  If nullptr, no tensors were
    // non-differentiable.
    PyObject *non_differentiable;
    // Python tuple of tensors which had inplace updates in the forward()
    // pass.  Set by Python with 'mark_dirty'.  If nullptr, no tensors were
    // modified inplace.
    PyObject *dirty_tensors;

    std::vector<torch::autograd::VariableInfo> output_info;
    std::vector<torch::autograd::VariableInfo> input_info;
    std::vector<torch::autograd::SavedVariable> saved_variables;
    // For each input, true if the input is a THPVariable
    std::vector<bool> is_variable_input;
    char has_freed_buffers;

    // The actual PyNode (in the autograd graph) that this data was
    // saved for.  This field may be NULL (because a user can construct
    // a THPFunction directly from Python), but when this field is non-NULL,
    // it is guaranteed that cdata.lock()->obj == this
    //
    // In most ordinary use, this field should always be non-NULL; e.g.,
    // when we allocate a THPFunction because we are running Node.apply,
    // after constructing a THPFunction, we immediately allocate a PyNode
    // for it.  We can't enforce this directly in the constructor of
    // THPFunction though, because there's no way to keep it live long enough
    // to save an owning reference to PyNode into the grad_fn of a Variable.
    std::weak_ptr<torch::autograd::PyNode> cdata;
};

bool THPFunction_initModule(PyObject *module);
extern PyTypeObject THPFunctionType;
extern PyObject *THPFunctionClass;

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}
