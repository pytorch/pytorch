#pragma once

#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <memory>

namespace caffe2 {
namespace serialize {
class PyTorchStreamReader;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    std::shared_ptr<StorageContext> storage_context = nullptr);

bool check_zip_file(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
