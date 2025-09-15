// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/script.h>

namespace py = pybind11;

namespace {

c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  py::object pyModule =
      py::module_::import("torch.backends._coreml.preprocess");
  py::object pyMethod = pyModule.attr("preprocess");

  py::dict modelDict =
      pyMethod(mod, torch::jit::toPyObject(method_compile_spec));

  c10::Dict<std::string, std::string> modelData;
  for (auto item : modelDict) {
    modelData.insert(
        item.first.cast<std::string>(), item.second.cast<std::string>());
  }
  return modelData;
}

static auto pre_reg =
    torch::jit::backend_preprocess_register("coreml", preprocess);

} // namespace
