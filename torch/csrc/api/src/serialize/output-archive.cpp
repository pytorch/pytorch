#include <torch/serialize/output-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/api/module.h>

#include <c10/util/Exception.h>

#include <memory>
#include <ostream>
#include <string>

namespace torch {
namespace serialize {
OutputArchive::OutputArchive(std::shared_ptr<jit::CompilationUnit> cu)
    : cu_(std::move(cu)),
      module_("__torch__.Module", cu_, /*shouldMangle=*/true) {}

void OutputArchive::write(const std::string& key, const c10::IValue& ivalue) {
  module_.register_attribute(key, ivalue.type(), ivalue);
}

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

void OutputArchive::save_to(
    const std::function<size_t(const void*, size_t)>& func) {
  jit::ExportModule(module_, func);
}
} // namespace serialize
} // namespace torch
