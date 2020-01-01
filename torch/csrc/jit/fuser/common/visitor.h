#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/fuser/common/ir.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API SimpleHandler {
  int handle(const Statement* statement);
  int handle(const Float* f);
};

struct TORCH_API IRPrinter {
  int handle(const Statement* statement);
  int handle(const Float* f);
  int handle(const Add* add);
};

}}} // torch::jit::fuser
