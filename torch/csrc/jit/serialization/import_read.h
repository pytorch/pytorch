#pragma once

#include <torch/csrc/api/include/torch/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <memory>

namespace torch::jit {

TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    std::optional<TypeResolver> type_resolver,
    std::optional<ObjLoader> obj_loader,
    std::optional<at::Device> device,
    torch::serialize::PyTorchStreamReader& stream_reader,
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser,
    std::shared_ptr<DeserializationStorageContext> storage_context = nullptr);

bool check_zip_file(
    const std::shared_ptr<torch::serialize::ReadAdapterInterface>& rai);

} // namespace torch::jit
