#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch {
namespace jit {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
#ifdef FBCODE_CAFFE2
  if (this->type()->getForwardHooks().size() > 0 ||
      this->type()->getForwardPreHooks().size() > 0) {
    throw std::runtime_error(
        "Cannot save module '" + this->type()->name()->name() +
        "' because it has forward hooks or pre-hooks attached. " +
        "Saving modules with hooks not supported in FBCODE yet. Please " +
        "remove the hooks before scripting if you want to save this model.");
  }
#endif
  ExportModule(*this, out, extra_files, false /* bytecode_format */);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
#ifdef FBCODE_CAFFE2
  if (this->type()->getForwardHooks().size() > 0 ||
      this->type()->getForwardPreHooks().size() > 0) {
    throw std::runtime_error(
       "Cannot save module '" + this->type()->name()->name() +
       "' because it has forward hooks or pre-hooks attached. " +
       "Saving modules with hooks not supported in FBCODE yet. Please " +
       "remove the hooks before scripting if you want to save this model.");
  }
#endif
  ExportModule(*this, filename, extra_files, false /* bytecode_format */);
}

void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      out,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info);
}

void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      filename,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info);
}

} // namespace jit
} // namespace torch
