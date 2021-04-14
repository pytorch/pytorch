#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch {
namespace jit {

// When bytecode_version is empty, bytecode won't be exported.
void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(
      *this, out, extra_files, at::optional<uint64_t>() /* bytecode version*/);
}

// When bytecode_version is empty, bytecode won't be exported.
void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(
      *this,
      filename,
      extra_files,
      at::optional<uint64_t>() /* bytecode version*/);
}

// bytecode will be exported, if version is a valid and supported number.
void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    at::optional<uint64_t> bytecode_version,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      out,
      extra_files,
      bytecode_version /* bytecode version */,
      save_mobile_debug_info);
}

// bytecode will be exported, if version is a valid and supported number.
void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    at::optional<uint64_t> bytecode_version,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      filename,
      extra_files,
      bytecode_version /* bytecode version*/,
      save_mobile_debug_info);
}

} // namespace jit
} // namespace torch
