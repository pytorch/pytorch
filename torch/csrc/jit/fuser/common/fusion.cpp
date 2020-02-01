#include <torch/csrc/jit/fuser/common/fusion.h>

namespace torch {
namespace jit {
namespace fuser {

thread_local Fusion* FusionGuard::cur_fusion = nullptr;

std::ostream& operator<<(std::ostream& os, const std::deque<Val*>& vals) {
  os << "( ";
  for (auto* val : vals) {
    os << val;
    if (val != *(vals.end()))
      os << ", ";
  }
  os << " )";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Fusion& fusion) {
  os << "Fusion{";
  if(fusion.inputs().size()>0)
       os<< "\n->Inputs " << fusion.inputs();

  if(fusion.exprs().size()>0){
    os << "\n->Body(\n";

    for (auto* expr : fusion.exprs()) {
      os << expr;
    }
    os<<"\n)";
  }

  if(fusion.outputs().size()>0)
    os << "\n->Outputs " << fusion.outputs();

  os<<"\n}";
  return os;
}

}}} // torch::jit::fuser
