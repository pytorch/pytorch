#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {

class TORCH_API StaticRuntime {
 public:
  StaticRuntime(std::shared_ptr<torch::jit::Graph> g) : graph_(std::move(g)) {}
  StaticRuntime(
      const torch::jit::Module& m,
      std::shared_ptr<torch::jit::Graph> g);

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps) const;

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::Module module_;
};

} // namespace jit
} // namespace torch
