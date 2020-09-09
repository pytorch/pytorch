
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace codegen {

std::string generateCudaKernel(
    const Kernel* kernel,
    const std::string& kernel_name) {
  const auto& allocations = kernel->globalAllocations();
  std::vector<Val*> global_tensors(allocations.size());
  std::transform(
      allocations.begin(),
      allocations.end(),
      global_tensors.begin(),
      [](kir::Allocate* alloc) { return alloc->buffer(); });

  std::stringstream ss;

  IRPrinter ir_printer(ss);
  ir_printer.printKernel(
      kernel->exprs(),
      kernel_name,
      global_tensors,
      !kernel->dynamicAllocations().empty());

  return ss.str();
}

} // namespace codegen
} // namespace fuser
} // namespace jit
} // namespace torch
