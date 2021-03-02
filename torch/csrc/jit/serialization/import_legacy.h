#pragma once

#include <torch/csrc/jit/api/module.h>

namespace caffe2 {
namespace serialize {
class PyTorchStreamReader;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

struct CompilationUnit;

// Deserializes a model in legacy format.
Module LEGACY_deserialize(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    const c10::optional<c10::Device>& device);

} // namespace jit
} // namespace torch
