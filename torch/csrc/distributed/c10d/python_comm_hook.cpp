#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include <torch/csrc/distributed/c10d/reducer.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace c10d {
PythonCommHook::PythonCommHook(py::object state, py::object hook)
    : state_(std::move(state)), hook_(std::move(hook)){};

c10::intrusive_ptr<torch::jit::Future> PythonCommHook::runHook(
    const GradBucket& bucket) {
  py::gil_scoped_acquire acquire;

  py::object py_fut = hook_(state_, bucket);

  try {
    return py_fut.cast<std::shared_ptr<torch::jit::PythonFutureWrapper>>()->fut;
  } catch (const py::cast_error& e) {
    auto type = py_fut.get_type();
    auto errMsg = c10::str(
        e.what(),
        ". DDP communication hook's callback must return a "
        "torch.futures.Future or torch._C.Future object, but got ",
        type.attr("__module__").cast<std::string>(),
        ".",
        type.attr("__qualname__").cast<std::string>());
    throw std::runtime_error(errMsg);
  }
}

std::vector<at::Tensor> PythonCommHook::processFuture(
    c10::IValue future_value) {
  // Since we have a Python hook, future_value can be a PyObject.
  if (future_value.isPyObject()) {
    // We first convert it to an IValue that contains a TensorVector.
    py::gil_scoped_acquire ag;
    py::object obj = torch::jit::toPyObject(future_value);
    auto value = torch::jit::toIValue(
        obj, c10::ListType::create(c10::TensorType::get()));

    return value.toTensorVector();
  }

  return future_value.toTensorVector();
}

} // namespace c10d
