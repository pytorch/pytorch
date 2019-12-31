#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/fuser/common/ir.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API SimpleHandler {
  int handle(Statement* statement);
  int handle(Float* f);
};

struct TORCH_API IRPrinter {
  int handle(Statement* statement);
  int handle(Float* f);
  int handle(Add* add);
};



}}} // torch::jit::fuser
