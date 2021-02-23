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
// into a mobile::Module object. For the overloads that accept
// the extra_files map, passing in a map with the special key
// named "$__get_all_extra__$" causes the method to load all
// extra files and fill up extra_files with their contents.
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

/**
 * Load only the contents of the "extra/" files whose names are
 * passed in the map (extra_files). Populate the corresponding values
 * with the contents of those files. Do not attempt to load the entire
 * model, and stop once the extra files have been extracted.
 *
 * This API is needed to be able to load GPU models on linux CPU
 * machines and extract only the extra files so that we can inspect
 * the metadata that was added to the .ptl archive when it was
 * generated.
 *
 */
void _load_extra_only_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files);

std::vector<std::string> _get_all_archive_file_names(
    const std::string& filename
);

std::vector<std::string> _get_all_archive_file_names(
    std::istream& in
);

} // namespace jit
} // namespace torch
