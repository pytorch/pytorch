#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

#include <istream>

namespace caffe2 {
namespace serialize {
class ReadAdapterInterface;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

static script::ExtraFilesMap default_extra_files;

TORCH_API script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

TORCH_API script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

TORCH_API script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `script::Module` from the given `istream`.
///
/// The istream must contain a serialized `script::Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API script::Module load(
    std::istream& in,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `script::Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `script::Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API script::Module load(
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

/// Loads a serialized `script::Module` from the given `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `script::Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API script::Module load(
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    c10::optional<c10::Device> device = c10::nullopt,
    script::ExtraFilesMap& extra_files = default_extra_files);

} // namespace jit
} // namespace torch
