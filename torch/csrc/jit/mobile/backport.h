#pragma once

#include <caffe2/serialize/file_adapter.h>

namespace torch {
namespace jit {

TORCH_API int64_t _get_runtime_bytecode_version();

} // namespace jit
} // namespace torch
