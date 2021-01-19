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
using ExtraFilesMap = std::unordered_map<std::string, std::string>;
static ExtraFilesMap default_extra_files_mobile;

// The family of methods below convery a serialized Mobile Module
// into a mobile::Module object.
TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files_mobile);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files_mobile);

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files_mobile);
} // namespace jit
} // namespace torch
