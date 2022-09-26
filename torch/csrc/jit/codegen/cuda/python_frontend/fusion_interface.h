#pragma once
#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

//! nvFuser Fusion IR namespace abbreviation
namespace Nvf = torch::jit::fuser::cuda;

namespace nvfuser {

//! \class FusionInterface
//! \brief Implements an Interface that represents an nvFuser IR object in
//! in python.
//!
//! Example 1 - Define fusion:
//!
//!   fs = Fusion()
//!   with FusionDefinition(fs) as fd :
//!       t0 = fd.define_tensor(3)
//!       s1 = fd.define_constant(3.)
//!       t2 = fd.ops.add(t0, s1)
//!       fd.add_output(t2)
//!
//!   input = torch.ones(2, 4, 8, device='cuda')
//!   for _ in range(5) :
//!      outputs = fs.execute([input])
//!
//! Example 2 - Use cached fusion, directly, based on id:
//!
//!   fs = Fusion(fusion_id)
//!
//!   input = torch.ones(2, 4, 8, device='cuda')
//!   for _ in range(5) :
//!      outputs = fs.execute([input])

class TORCH_CUDA_CU_API FusionInterface {
 public:
  //! Pybind11 cannot bind to c10::optional and Pytorch is compiled with C++14.
  //! Therefore, I am adding two constructors, instead.
  FusionInterface();
  FusionInterface(size_t fusion_id);

  //! Define which Fusion IR object the interface represents
  void define(size_t fusion_id);
  //! Query whether the Fusion IR is defined
  bool defined() const;
  //! Return fusion id of this Fusion
  size_t id() const;

  //! Adds an input to the represented Fusion IR.
  void addInput(Nvf::Val* input) const;
  //! Adds an Output to the represented Fusion IR.
  void addOutput(Nvf::Val* output) const;
  //! Executes a fusion if the current cache pointer points at a terminal node
  std::vector<at::Tensor> execute(
      const at::ArrayRef<c10::IValue>& inputs) const;
  //! Activates a guard around the represented Fusion IR.
  Nvf::FusionGuard guard() const;
  //! Prints the represented nvFuser IR
  void print() const;

 private:
  //! Provides a pointer to the FusionExecutorCache that maps the current
  //! unscheduled Fusion IRs to scheduled Fusion IRs for execution.
  Nvf::FusionExecutorCache* fusionExecutorCachePtr() const;
  //! Points to the nvFuser Fusion IR object
  Nvf::Fusion* fusionPtr() const;

  c10::optional<size_t> fusion_id_;
};

} // namespace nvfuser
