#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include <ATen/core/functional.h>
#include <c10d/reducer.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/tensor_flatten.h>

namespace c10d {

PythonCommHook::~PythonCommHook() {
  py::gil_scoped_acquire ag;
  state_.dec_ref();
  hook_.dec_ref();
  // Explicitly set state_ and hook_ to nullptr to prevent py::object's dtor
  // to decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  state_.ptr() = nullptr;
  hook_.ptr() = nullptr;
}

c10::intrusive_ptr<c10::ivalue::Future> PythonCommHook::runHook(
    GradBucket& bucket) {
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

std::vector<at::Tensor> PythonCommHook::parseHookResult(
    const c10::IValue& result) {
  TORCH_INTERNAL_ASSERT(
      result.isPyObject() || result.isTensorList(),
      "expected the hook result is either a PyObject or TensorList");

  if (result.isPyObject()) {
    py::gil_scoped_acquire ag;
    py::object obj = torch::jit::toPyObject(result);
    auto value = torch::jit::toIValue(
        obj, c10::ListType::create(c10::TensorType::get()));

    return value.toTensorVector();
  }

  return result.toTensorVector();
}

} // namespace c10d
