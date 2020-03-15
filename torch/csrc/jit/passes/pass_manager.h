#pragma once

#include <torch/csrc/jit/ir/ir.h>

/* `getCustomPrePasses()` returns a vector of passes that will be executed
 * after differentiation but before any fusion. This is the de-facto location
 * for compiler backends to insert passes.
 *
 * `getCustomPostPasses()` returns a vector of passes that will be
 * executed after differentiation and after fusion (if any). This is the
 * location for fusion cleanup passes if they are needed.
 *
 * Static registration of a pass can be done by creating a global
 * `Register{Pre,Post}Pass r(Pass)` variable in a compilation unit.
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

TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPostPasses();
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPrePasses();

struct TORCH_API RegisterPostPass {
  // Back-compat
  RegisterPostPass(GraphPass p);
  static GraphPassNameType registerPostPass(GraphPass p);
};

using RegisterPass = RegisterPostPass;

struct TORCH_API RegisterPrePass {
  // Back-compat
  RegisterPrePass(GraphPass p);
  static GraphPassNameType registerPrePass(GraphPass p);
};

struct TORCH_API ClearPostPass {
  ClearPostPass(GraphPassNameType p);
};

struct TORCH_API ClearPrePass {
  ClearPrePass(GraphPassNameType p);
};

struct TORCH_API ClearAllPostPasses {
  ClearAllPostPasses();
};

struct TORCH_API ClearAllPrePasses {
  ClearAllPrePasses();
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
