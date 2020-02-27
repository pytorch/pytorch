
copy: fbcode/caffe2/torch/csrc/jit/serialization/import_legacy.h
copyrev: e8fb519d2f7dedef18480af9b75d64524040ea49

#pragma once

#include <torch/csrc/jit/api/module.h>

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
