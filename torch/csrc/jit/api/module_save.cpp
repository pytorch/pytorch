#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

C10_DEFINE_bool(
  torch_jit_serialize_modules_with_hooks_fbcode,
  false,
  "Allows modules with hooks to be serialized in FBCODE. WARNING: only "
  "enable after you verify the PyTorch binary which will load the "
  "serialized modules with hooks is older than 1/21/2021. Otherwise the "
  "loading binary will throw a Runtime error.");

namespace torch {
namespace jit {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
#ifdef FBCODE_CAFFE2
  if ((this->type()->getForwardHooks().size() > 0 ||
      this->type()->getForwardPreHooks().size() > 0) &&
      !FLAGS_torch_jit_serialize_modules_with_hooks_fbcode) {
    throw std::runtime_error(
        "Cannot save module '" + this->type()->name()->name() +
        "' because it has forward hooks or pre-hooks attached. " +
        "To enable serializing modules with hooks attached, set " +
        "flag 'torch_jit_serialize_modules_with_hooks_fbcode' " +
        "to true after verifying that the PyTorch binary which is to load " +
        "the module with serizliaed hooks is older than 1/21/2021. ");
  }
#endif
  ExportModule(*this, out, extra_files, false /* bytecode_format */);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
#ifdef FBCODE_CAFFE2
  if ((this->type()->getForwardHooks().size() > 0 ||
      this->type()->getForwardPreHooks().size() > 0) &&
      !FLAGS_torch_jit_serialize_modules_with_hooks_fbcode) {
    throw std::runtime_error(
        "Cannot save module '" + this->type()->name()->name() +
        "' because it has forward hooks or pre-hooks attached. " +
        "To enable serializing modules with hooks attached, set " +
        "flag 'torch_jit_serialize_modules_with_hooks_fbcode' " +
        "to true after verifying that the PyTorch binary which is to load " +
        "the module with serizliaed hooks is older than 1/21/2021. ");
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
