#pragma once
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

#include <istream>
#include <memory>

#include <caffe2/serialize/file_adapter.h>

namespace torch {
namespace jit {
using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

constexpr const char* kArchiveNameBytecode = "bytecode";
constexpr const char* kArchiveNameConstants = "constants";
constexpr const char* kArchiveNameVersion = "version";

// The family of methods below load a serialized Mobile Module
// into a mobile::Module object.
TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_file,
    uint64_t module_load_options = kDefaultMobileLoadOptions);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files);

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options = kDefaultMobileLoadOptions);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options);

TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device = c10::nullopt);

TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device = c10::nullopt);

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device = c10::nullopt);

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
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files);

// Currently used by both mobile/import.cpp and model_compatibility.cpp.
// Should be removed after model_compatibility.cpp start using simplified
// version type_resolver and obj_loader.
at::TypePtr resolveTypeNameMobile(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit);
c10::StrongTypePtr typeResolverMobile(
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit);
c10::intrusive_ptr<c10::ivalue::Object> objLoaderMobile(
    const at::StrongTypePtr& type,
    const at::IValue& input,
    mobile::CompilationUnit& mobile_compilation_unit);

// Given a reader, which has access to a model file,
// return true if there exists tensors in `bytecode` archive
bool isTensorInBytecodeArchive(
    caffe2::serialize::PyTorchStreamReader& stream_reader);

namespace mobile {

/**
 * Given a torch::jit::mobile::Module, return a set of operator names
 * (with overload name) that are used by any method in this mobile
 * Mobile. This method runs through the bytecode for all methods
 * in the specified model (module), and extracts all the root
 * operator names. Root operators are operators that are called
 * directly by the model (as opposed to non-root operators, which
 * may be called transitively by the root operators).
 *
 */
TORCH_API std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module);

} // namespace mobile

} // namespace jit
} // namespace torch
