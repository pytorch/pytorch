#pragma once

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/unpickler.h>

#include <istream>

namespace caffe2 {
namespace serialize {
class ReadAdapterInterface;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

// For reading unified serialization format from torch.Package
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    std::shared_ptr<torch::jit::DeserializationStorageContext> storage_context,
    c10::optional<at::Device> device,
    std::string ts_id /* torchscript identifier inside package */);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

/// Loads a serialized `Module` from the given `istream`.
///
/// The istream must contain a serialized `Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API Module
load(std::istream& in, c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module load(
    std::istream& in,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module load(
    const std::string& filename,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

/// Loads a serialized `Module` from the given shared_ptr `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt);

TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version);

extern Module (*_load_jit_module_from_flatbuffer_bytes)(
    // comp unit
    std::shared_ptr<char>,
    size_t,
    ExtraFilesMap&,
    c10::optional<at::Device>);

extern Module (*_load_jit_module_from_flatbuffer_bytes)(
    // comp unit
    std::shared_ptr<char>,
    size_t,
    ExtraFilesMap&,
    c10::optional<at::Device>);

} // namespace jit
} // namespace torch
