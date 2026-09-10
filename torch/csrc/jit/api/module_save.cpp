#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch::jit {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, false /* bytecode_format */);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(*this, filename, extra_files, false /* bytecode_format */);
}

void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    bool use_flatbuffer) const {
  ExportModule(
      *this,
      out,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info,
      use_flatbuffer);
}

void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    bool use_flatbuffer) const {
  ExportModule(
      *this,
      filename,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info,
      use_flatbuffer);
}

} // namespace torch::jit
