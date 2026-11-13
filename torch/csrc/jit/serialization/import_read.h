#pragma once

#include <torch/csrc/jit/serialization/unpickler.h>
#include <memory>

namespace caffe2::serialize {
class PyTorchStreamReader;
} // namespace caffe2::serialize

namespace torch::jit {

TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    std::optional<TypeResolver> type_resolver,
    std::optional<ObjLoader> obj_loader,
    std::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser,
    std::shared_ptr<DeserializationStorageContext> storage_context = nullptr);

bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai);

} // namespace torch::jit
