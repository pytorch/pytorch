#pragma once
#include <torch/csrc/jit/mobile/module.h>

#include <istream>
#include <memory>

#include <caffe2/serialize/file_adapter.h>

namespace torch {
namespace jit {
using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

// The family of methods below load a serialized Mobile Module
TORCH_API void _backport_for_mobile(std::istream& in);

TORCH_API void _backport_for_mobile(const std::string& filename);

TORCH_API void _backport_for_mobile(std::unique_ptr<ReadAdapterInterface> rai);

//// The family of methods below to get version given bytecode model
// TORCH_API void _get_version_for_mobile(
//    std::istream& in);
//
// TORCH_API void _get_version_for_mobile(
//    const std::string& filename);
//
// TORCH_API void _get_version_for_mobile(
//    std::unique_ptr<ReadAdapterInterface> rai);

} // namespace jit
} // namespace torch
