#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/onnx/onnx.h>
#include <torch/csrc/jit/mobile/module.h>

#include <ostream>

namespace torch {
namespace jit {
namespace mobile {

TORCH_API void ExportModule(
    const torch::jit::mobile::Module& module,
    std::ostream& out,
    bool bytecode_format = false);

TORCH_API void ExportModule(
    const torch::jit::mobile::Module& module,
    const std::string& filename,
    bool bytecode_format = false);

TORCH_API void ExportModule(
    const torch::jit::mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    bool bytecode_format = false);

} // namespace mobile
} // namespace jit
} // namespace torch
