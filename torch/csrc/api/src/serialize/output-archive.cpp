#include <torch/serialize/output-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/script/module.h>

#include <c10/util/Exception.h>

#include <memory>
#include <ostream>
#include <string>

namespace torch {
namespace serialize {
OutputArchive::OutputArchive()
    : module_("__main__") {}

void OutputArchive::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  module_.register_parameter(key, tensor, is_buffer);
}

void OutputArchive::write(
    const std::string& key,
    OutputArchive& nested_archive) {
  module_.register_module(key, nested_archive.module_);
}

void OutputArchive::save_to(const std::string& filename) {
  jit::ExportModule(module_, filename);
}

void OutputArchive::save_to(std::ostream& stream) {
  jit::ExportModule(module_, stream);
}
} // namespace serialize
} // namespace torch
