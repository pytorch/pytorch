#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/ir_printer.h>
#include <torch/csrc/jit/fuser/common/type.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

namespace torch {
namespace jit {
namespace fuser {

void IRPrinter::print(const Fusion* const fusion){
  irstream_ << "\nPrinting TensorViews...\n";
  for(auto &val : fusion->vals()) {
    if(val->getValType().value() == ValType::TensorView)
      irstream_ << "\t" << cnt_++ << " " << val << "\n";
  }
  
  cnt_ = 0;   
  irstream_ << "\nPrinting Operator Expressions...\n";
  traverse(fusion, false /*from_outputs_only*/, {ValType::TensorView}, false /*breadth_first*/);

  cnt_ = 0; 
  irstream_ << "\nPrinting Tensor Expressions...\n";
  traverse(fusion, false /*from_outputs_only*/, {ValType::TensorDomain}, false /*breadth_first*/);
  irstream_ << "\n";
}

} // namespace fuser
} // namespace jit
} // namespace torch
