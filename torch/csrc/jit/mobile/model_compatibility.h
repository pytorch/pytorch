#pragma once

#include <istream>
#include <memory>
#include <unordered_map>

namespace caffe2 {
namespace serialize {
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

// Struct storing metadata of an operator that can be useful for versioning
struct OperatorInfo {
  // The number of arguments within the schema of the op
  // default to -1 if no schema information is available as that value
  // cannot exist in a well formed schema
  int num_schema_args = -1;
};

// The family of methods below to get the root ops and information from a model
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
