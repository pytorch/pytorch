#pragma once

#include <ATen/ATen.h>
#include <c10/core/SafePyObject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch::autograd {

// Copy/move are already deleted on SavedVariableHooks; clang-tidy doesn't
// see that inherited deletion, hence the suppression below.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct PySavedVariableHooks : public SavedVariableHooks {
  PySavedVariableHooks(py::function&& pack_hook, py::function&& unpack_hook);
  void call_pack_hook(const at::Tensor& tensor) override;
  at::Tensor call_unpack_hook() override;
  ~PySavedVariableHooks() override {
    // Batch the three SafePyObject decrefs under one GIL acquisition.
    py::gil_scoped_acquire gil;
    pack_hook_.reset();
    unpack_hook_.reset();
    data_.reset();
  }
  std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  retrieve_unpack_hook_data() const override;
  std::optional<c10::SafePyObject> retrieve_unpack_hook() const override;

 private:
  const c10::SafePyObject& data() const {
    TORCH_CHECK(
        data_.has_value(),
        "the pack hook raised before saving data for this tensor");
    return *data_;
  }

  std::optional<c10::SafePyObject> pack_hook_;
  std::optional<c10::SafePyObject> unpack_hook_;
  std::optional<c10::SafePyObject> data_;
};

struct PyDefaultSavedVariableHooks {
  static void push_hooks(py::function& pack_hook, py::function& unpack_hook);
  static void pop_hooks();
  static std::unique_ptr<SavedVariableHooks> get_hooks();
};

} // namespace torch::autograd
