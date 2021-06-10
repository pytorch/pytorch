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

// used in torch.package deserialization
class TORCH_API StorageContext {
 public:
  explicit StorageContext() : unique_id(0) {};

  uint64_t addStorage(const std::string& name, c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    uint64_t id = unique_id++; 
    cdata_id_map_.insert({name, id});
    id_storage_map_.insert({id, storage});
    return id;
  }

  bool hasStorage(const std::string& name) {
    return cdata_id_map_.find(name) != cdata_id_map_.end();
  }

  bool getStorageID(const std::string& name) {
      TORCH_INTERNAL_ASSERT(hasStorage(name));
      return cdata_id_map_.find(name)->second;
  }

  c10::Storage getStorage(const std::string& name) {
    TORCH_INTERNAL_ASSERT(hasStorage(name));
    return id_storage_map_.find(cdata_id_map_.find(name)->second)->second;
  }
  ~StorageContext() = default;

 private:
  uint64_t unique_id;
  std::map<std::string, uint64_t> cdata_id_map_;
  std::map<uint64_t, c10::Storage> id_storage_map_; 
};

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
    std::shared_ptr<torch::jit::StorageContext> storage_context,
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

} // namespace jit
} // namespace torch
