#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>

#include <stack>

//Could be in a .cpp file:
#include <torch/csrc/jit/fuser/common/fusion.h>

//Not needed:
#include <torch/csrc/jit/fuser/common/iriostream.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API TransformReplay : public IterVisitor {

  //Trace back the history of td, record the Expr's that made this td (split, merge, reorder)
  static const TensorDomain* get_root(const TensorDomain* td, std::stack<const Expr*> *record = nullptr){
    const TensorDomain *root = td;
    Fusion* fusion = FusionGuard::getCurFusion();
    //Get my origin
    const Expr* orig = fusion->origin(td);
    //If I'm not back to the original td
    while(orig != nullptr){
  
      const TensorDomain* previous_td = nullptr;
      //Check inputs of this operation, make sure there isn't more than one TD
      //I can only record operations that only take this TD as an input.
      for(const Val* inp : orig->inputs())
        if(inp->getValType() == ValType::TensorDomain){
          if(previous_td != nullptr)
            throw std::runtime_error("TransformReplay::get_root could not decifer transform history of a TensorDomain.");
          //Traverse back
          previous_td = static_cast<const TensorDomain*>(inp);
          record->push(orig);
        }
      root = previous_td;
      orig = fusion->origin(root);
    }
    return root;
  }

  /* 
   * Takes replay_ref and replays its transformations on replay_target
   * Replays from begining of both TensorDomains. could be more efficient to try and find a common ancestor
   * to start from
   */
  static const TensorView* replay(const TensorView* replay_target, const TensorView* replay_ref){
    const TensorView* target_root = replay_target;
    const TensorView* ref_root = replay_ref;

  }


};

}}}