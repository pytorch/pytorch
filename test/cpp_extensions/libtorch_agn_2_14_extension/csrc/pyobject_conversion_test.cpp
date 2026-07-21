#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>

// Exercises the libtorch-only PyObject<->Tensor stable shims. The extension
// links only libtorch; the shims dispatch through a vtable that libtorch_python
// registers, which is present here because the test process has imported torch.
//
// The shims require the GIL. A boxed STABLE_TORCH_LIBRARY kernel may run with
// the GIL released (the dispatcher does so for backward, torch.compile, and
// intra-op threads), so the kernel must re-acquire it with PyGILState_Ensure
// before touching Python objects -- the same pattern a real extension would use.

using torch::stable::Tensor;

// stable Tensor -> PyObject -> stable Tensor; the result shares storage with
// the input. Exercises both to_pyobject and from_pyobject.
Tensor my_pyobject_roundtrip(Tensor t) {
  PyGILState_STATE gil = PyGILState_Ensure();
  void* py = torch::stable::to_pyobject(t);
  Tensor out = torch::stable::from_pyobject(py);
  Py_DECREF(static_cast<PyObject*>(py)); // to_pyobject returns a new reference
  PyGILState_Release(gil);
  return out;
}

// Like above but does real work on the tensor obtained from from_pyobject.
Tensor my_pyobject_sum(Tensor t) {
  PyGILState_STATE gil = PyGILState_Ensure();
  void* py = torch::stable::to_pyobject(t);
  Tensor out = torch::stable::sum(torch::stable::from_pyobject(py));
  Py_DECREF(static_cast<PyObject*>(py));
  PyGILState_Release(gil);
  return out;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_pyobject_roundtrip(Tensor t) -> Tensor");
  m.def("my_pyobject_sum(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_pyobject_roundtrip", TORCH_BOX(&my_pyobject_roundtrip));
  m.impl("my_pyobject_sum", TORCH_BOX(&my_pyobject_sum));
}
