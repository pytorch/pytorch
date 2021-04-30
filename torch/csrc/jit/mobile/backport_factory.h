#pragma once

//#include <ATen/core/ivalue.h>
#include <memory>

namespace c10 {
struct IValue;
}

namespace caffe2 {
namespace serialize {
class ReadAdapterInterface;
class PyTorchStreamWriter;
class PyTorchStreamReader;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

constexpr const char* kArchiveNameConstants = "constants";
constexpr const char* kArchiveNameBytecode = "bytecode";
constexpr int64_t kBytecodeVersionV4 = 0x4L;

// The family of methods below backport one model vn to vn-1
bool backport_v5_to_v4(
    caffe2::serialize::PyTorchStreamReader& rai,
    caffe2::serialize::PyTorchStreamWriter& writer,
    std::vector<c10::IValue>& bytecode_values);

// The family of methods is common for most backport functions.
// They are located in backport_common.cpp
bool update_bytecode_version(
    std::vector<c10::IValue>& bytecode_values,
    const int64_t to_version);

void copy_non_bytecode(
    caffe2::serialize::PyTorchStreamReader& reader,
    caffe2::serialize::PyTorchStreamWriter& writer);

} // namespace jit
} // namespace torch
