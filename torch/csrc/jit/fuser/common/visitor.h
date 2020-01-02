#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API SimpleHandler {
  int handle(const Statement* statement);
  int handle(const Float* f);
};

struct TORCH_API IRPrinter {
  virtual ~IRPrinter() = default;
  IRPrinter() = delete;
  IRPrinter(
    std::ostream& _out)
  : out_{_out} { }

  IRPrinter(const IRPrinter& other) = default;
  IRPrinter& operator=(const IRPrinter& other) = default;

  IRPrinter(IRPrinter&& other) = default;
  IRPrinter& operator=(IRPrinter&& other) = default;

  int handle(const Statement* statement);
  int handle(const Float* f);
  int handle(const Add* add);

private:
  std::ostream& out_;
};

}}} // torch::jit::fuser
