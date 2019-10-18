#pragma once

#include <torch/csrc/jit/script/module.h>

namespace caffe2 {
namespace serialize {
class PyTorchStreamReader;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

namespace script {
struct CompilationUnit;
} // script

// Deserializes a model in legacy format.
script::Module LEGACY_deserialize(
    std::shared_ptr<script::CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    const c10::optional<c10::Device>& device);

} // namespace jit
} // namespace torch
