#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/export.h>

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

// Dump all operator names of the module and submodules in yaml format.
void Module::dump_op_names(const std::string& filename) const {
  std::unordered_set<std::string> opnames;
  torch::jit::export_opnames(*this, opnames);
  std::ofstream ofile(filename);
  for (const auto& name : opnames) {
    ofile << "- " << name << std::endl;
  }
  ofile.close();
}
} // namespace script
} // namespace jit
} // namespace torch
