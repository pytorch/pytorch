#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>

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
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_bytecode_version(std::istream& in);

TORCH_API uint64_t _get_model_bytecode_version(const std::string& filename);

TORCH_API uint64_t _get_model_bytecode_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

uint64_t _get_model_bytecode_version(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below to get the operator version from a model
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_operator_version(std::istream& in);

TORCH_API uint64_t _get_model_operator_version(const std::string& filename);

TORCH_API uint64_t _get_model_operator_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// Utility Functions
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

// The family of methods below to get contained types from a model
// Throws if not passed in a well formed model
TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::istream& in);

TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::string& filename);

TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below return the compatibility information of a model
struct ModelCompatibilityInfo {
  uint64_t bytecode_version;
  std::unordered_map<std::string, OperatorInfo> operator_info;
  std::unordered_set<std::string> type_table;
  uint64_t operator_version;

  // Factory Methods
  static TORCH_API ModelCompatibilityInfo get(std::istream& in);
  static TORCH_API ModelCompatibilityInfo get(const std::string& filename);
  static TORCH_API ModelCompatibilityInfo
  get(std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);
};

enum ModelCompatibilityStatus {
  OK = 1,
  ERROR = 2,
};

struct ModelCompatCheckResult {
  ModelCompatibilityStatus status;
  std::vector<std::string> errors;
};
// Takes in information about a runtime and a model and returns if the two are
// compatible with one another.
TORCH_API ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info,
    ModelCompatibilityInfo model_info);

} // namespace jit
} // namespace torch
