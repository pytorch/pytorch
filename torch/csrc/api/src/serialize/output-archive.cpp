#include <torch/serialize/output-archive.h>

#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/script/module.h>

#include <memory>
#include <string>

namespace torch {
namespace serialize {
OutputArchive::OutputArchive()
    : module_(std::make_shared<jit::script::Module>()) {}

void OutputArchive::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  module_->register_parameter(key, tensor, is_buffer);
}

void OutputArchive::write(
    const std::string& key,
    OutputArchive& nested_archive) {
  module_->register_module(key, nested_archive.module_);
}

void save_to_file(const OutputArchive& archive, const std::string& filename) {
  jit::ExportModule(*archive.module_, filename);
}
} // namespace serialize
} // namespace torch
