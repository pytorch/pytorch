#pragma once

#include <torch/csrc/jit/ir/ir.h>

/* `getCustomPreFusionPasses()` returns a vector of passes that will be executed
 * after differentiation but before any fusion. This is the de-facto location
 * for compiler backends to insert passes.
 *
 * `getCustomPostFusionPasses()` returns a vector of passes that will be
 * executed after differentiation and after fusion (if any). This is the
 * location for fusion cleanup passes if they are needed.
 *
 * Static registration of a pass can be done by creating a global
 * `Register{Pre,Post}FusionPass r(Pass)` variable in a compilation unit.
 *
 * pass_manager.h uses a Meyer's singleton to store a vector of `Pass`es, which
 * modify the IR graph in place.
 */

namespace torch {
namespace jit {

// A pass modifies a Graph in place.
using GraphPass = std::function<void(std::shared_ptr<Graph>&)>;
using GraphPassNameType = unsigned int;
using GraphPassEntry = std::pair<GraphPass, GraphPassNameType>;
static GraphPassNameType graphPassID = 1;

TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPostFusionPasses();
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPreFusionPasses();

struct TORCH_API RegisterPostFusionPass {
  // Back-compat
  RegisterPostFusionPass(GraphPass p);
  static GraphPassNameType registerPostFusionPass(GraphPass p);
};

using RegisterPass = RegisterPostFusionPass;

struct TORCH_API RegisterPreFusionPass {
  // Back-compat
  RegisterPreFusionPass(GraphPass p);
  static GraphPassNameType registerPreFusionPass(GraphPass p);
};

struct TORCH_API ClearPostFusionPass {
  ClearPostFusionPass(GraphPassNameType p);
};

struct TORCH_API ClearPreFusionPass {
  ClearPreFusionPass(GraphPassNameType p);
};

struct TORCH_API ClearAllPostFusionPasses {
  ClearAllPostFusionPasses();
};

struct TORCH_API ClearAllPreFusionPasses {
  ClearAllPreFusionPasses();
};

// Mechanism to be able to remove a registered pass
// Each pass needs to inherit this class as it's based on
// static members.
struct TORCH_API PassManager{
private:
  // Force class to be abstract
  virtual void abstract() = 0;
protected:
  static GraphPassNameType name(GraphPassNameType PassName = 0, bool set = false);
  static bool flipRegistered(bool flip = false);
public:
  static void registerPass(GraphPass p);
  static void clearPass();
};

} // namespace jit
} // namespace torch
