#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct TORCH_API BaseMutator {

  void mutate(Fusion* fusion);
  const Statement* mutate(const Statement* const statement);
  const Statement* mutate(const Float* const f);
  const Statement* mutate(const Int* const i);
  const Statement* mutate(const Add* const add);

};

}}} // torch::jit::fuser
