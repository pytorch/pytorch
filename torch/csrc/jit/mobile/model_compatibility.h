#pragma once

#include <istream>
#include <memory>

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

} // namespace jit
} // namespace torch
