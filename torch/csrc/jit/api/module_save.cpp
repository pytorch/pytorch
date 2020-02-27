
copy: fbcode/caffe2/torch/csrc/jit/api/module_save.cpp
copyrev: 5207f6ba4be86ce3f51317252f380acfd01b232d

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch {
namespace jit {
namespace script {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, false);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(*this, filename, extra_files, false);
}

void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, true);
}

void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files) const {
  ExportModule(*this, filename, extra_files, true);
}

} // namespace script
} // namespace jit
} // namespace torch
