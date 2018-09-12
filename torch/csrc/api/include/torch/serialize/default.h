#pragma once

#include <torch/serialize/base.h>
#include <torch/tensor.h>

#include <torch/csrc/jit/script/module.h>

#include <memory>
#include <string>

namespace torch {
namespace serialize {
class DefaultWriter : public Writer {
 public:
  explicit DefaultWriter(std::string filename);

  void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false) override;

  void finish();

 private:
  std::string filename_;
  jit::script::Module module_;
};

class DefaultReader : public Reader {
 public:
  explicit DefaultReader(std::string filename);

  void read(const std::string& key, Tensor& tensor, bool is_buffer = false)
      override;

 private:
  std::shared_ptr<jit::script::Module> module_;
};

} // namespace serialize
} // namespace torch
