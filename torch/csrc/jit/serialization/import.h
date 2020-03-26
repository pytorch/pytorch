#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>

#include <istream>

namespace caffe2 {
namespace serialize {
class ReadAdapterInterface;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

static ExtraFilesMap default_extra_files;

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `Module` from the given `istream`.
///
/// The istream must contain a serialized `Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `Module` from the given `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

} // namespace jit
} // namespace torch
