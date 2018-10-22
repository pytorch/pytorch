#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"

#include <memory>
#include <cstdint>
#include <vector>
#include <unordered_map>

 namespace torch { namespace jit { namespace fuser {
   
 // "Kernel Specification." - Contains device-independent fusion information.
struct KernelSpec {
  KernelSpec(
    const int64_t _key
  , std::shared_ptr<Graph> _graph)
  : key_{_key}
  , graph_{_graph} 
  , code_{_graph}
  , nInputs_{_graph->inputs().size()}
  { }

   // Getters
  int64_t key() const { return key_; }
  std::shared_ptr<Graph> graph() const { return graph_; }
  const Code& code() const { return code_; }
  int64_t nInputs() const { return nInputs_; }
  bool isFusable() const { return isFusable_; }

   // Setters
  void setFusable(const bool _isFusable) { isFusable_ = _isFusable; }

private: 
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  int64_t nInputs_;
  bool isFusable_ = true;
};

} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
