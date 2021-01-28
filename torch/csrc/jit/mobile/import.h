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

// The family of methods below load a serialized Mobile Module
// into a mobile::Module object.
TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device = c10::nullopt);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device = c10::nullopt);

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt);
} // namespace jit
} // namespace torch
