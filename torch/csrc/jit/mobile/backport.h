#pragma once

#include <istream>
#include <memory>

#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>

namespace torch {
namespace jit {
namespace mobile {

// The family of methods below load a serialized Mobile Module
bool _backport_for_mobile(std::istream& in, std::ostream& out);

bool _backport_for_mobile(std::istream& in, const std::string& output_filename);

bool _backport_for_mobile(const std::string& input_filename, std::ostream& out);

bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename);

bool _backport_for_mobile(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::shared_ptr<caffe2::serialize::PyTorchStreamWriter> writer);

} // namespace mobile
} // namespace jit
} // namespace torch
