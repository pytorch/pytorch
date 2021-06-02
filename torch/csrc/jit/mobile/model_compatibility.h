#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>

#include <istream>
#include <memory>
#include <unordered_map>

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

std::vector<c10::IValue> get_bytecode_ivalues(
    caffe2::serialize::PyTorchStreamReader& reader);

c10::IValue readArchive(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

bool check_zip_file(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// The family of methods below to get the root ops and information from a model
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
