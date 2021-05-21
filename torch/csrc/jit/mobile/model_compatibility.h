#pragma once

#include <istream>
#include <memory>

namespace caffe2 {
namespace serialize {
class PyTorchStreamReader;
class ReadAdapterInterface;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

// The family of methods below to get bytecode version from a model
TORCH_API int64_t _get_model_bytecode_version(std::istream& in);

TORCH_API int64_t _get_model_bytecode_version(const std::string& filename);

TORCH_API int64_t _get_model_bytecode_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

int64_t _get_model_bytecode_version(
    const std::vector<c10::IValue>& bytecode_ivalues);

std::vector<c10::IValue> get_bytecode_values(
    caffe2::serialize::PyTorchStreamReader& reader);

c10::IValue readArchive(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

bool check_zip_file(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
