#pragma once
#include <torch/csrc/jit/mobile/module.h>

#include <istream>
#include <memory>

#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>

namespace torch {
namespace jit {
using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

// The family of methods below to get version given bytecode model
TORCH_API int64_t _get_bytecode_version(std::istream& in);

TORCH_API int64_t _get_bytecode_version(const std::string& filename);

TORCH_API int64_t
_get_bytecode_version(std::shared_ptr<ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
