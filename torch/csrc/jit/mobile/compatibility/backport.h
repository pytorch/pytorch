#pragma once

#include <c10/macros/Export.h>
#include <istream>
#include <memory>

namespace caffe2::serialize {
class ReadAdapterInterface;
class PyTorchStreamWriter;
} // namespace caffe2::serialize

namespace torch::jit {

TORCH_API bool _backport_for_mobile(
    std::istream& in,
    std::ostream& out,
    const int64_t to_version);

TORCH_API bool _backport_for_mobile(
    std::istream& in,
    const std::string& output_filename,
    const int64_t to_version);

TORCH_API bool _backport_for_mobile(
    const std::string& input_filename,
    std::ostream& out,
    const int64_t to_version);

TORCH_API bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename,
    const int64_t to_version);

} // namespace torch::jit
