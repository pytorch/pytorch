#pragma once

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <istream>

namespace caffe2::serialize {
class ReadAdapterInterface;
} // namespace caffe2::serialize

namespace torch::jit {

class DeserializationStorageContext;

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

// For reading unified serialization format from torch.Package
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    std::shared_ptr<torch::jit::DeserializationStorageContext> storage_context,
    std::optional<at::Device> device,
    const std::string& ts_id /* torchscript identifier inside package */);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `istream`.
///
/// The istream must contain a serialized `Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given shared_ptr `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

TORCH_API Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version);

TORCH_API Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

TORCH_API Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

TORCH_API Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

// Equivalent to ObjLoaderFuncWithVersion with an unknown archive version, i.e.
// container type tags are always re-derived. Kept with this exact signature so
// it continues to convert to ObjLoader.
TORCH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input);

// `archive_version` is the file-format version of the archive being read, as
// reported by PyTorchStreamReader::version(). See [type tag serialization] in
// unpickler.h: for version >= 3 the archive carries container type strings and
// the unpickler applies them as each container is built, so re-deriving them
// here is unnecessary and can cost tens of seconds on a large model. Pass 0 if
// the version is unknown, which conservatively keeps the historical behaviour.
TORCH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFuncWithVersion(
    const at::StrongTypePtr& type,
    IValue input,
    uint64_t archive_version);

} // namespace torch::jit
