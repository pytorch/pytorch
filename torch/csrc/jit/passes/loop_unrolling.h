#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// return true if graph is modified
TORCH_API bool UnrollLoops(std::shared_ptr<Graph>& graph);

TORCH_API Node* PeelLoop(Node* n, size_t times);

// return true if graph is modified
TORCH_API bool PeelProfilingLoops(const std::shared_ptr<Graph>& graph);

struct TORCH_API LoopsPeeler {
  LoopsPeeler(std::function<bool(Node* n)> callback, size_t num_iterations = 1)
      : callback_(std::move(callback)), num_iterations_(num_iterations) {}

  bool run(const std::shared_ptr<Graph>& graph);

 private:
  void collectLoop(Node* n);
  void collectLoops(Block* block);
  void peelLoops();

  std::function<bool(Node* n)> callback_ = nullptr;
  Node* in_loop_ = nullptr;
  std::list<Node*> loops_to_peel_;
  size_t num_iterations_ = 1;
};
} // namespace jit
} // namespace torch
