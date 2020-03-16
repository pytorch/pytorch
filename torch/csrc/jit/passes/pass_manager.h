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

// Since Passes are std::functions, we associate a UUID to each pass, this way if we want to deregister a pass, we have something to reference it by.
using GraphPassNameType = unsigned int;
// Start UUID at 1
static GraphPassNameType graphPassID = 1;

<<<<<<< HEAD
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPostPasses();
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPrePasses();
=======
// Graph pass entries have a name associated with them
using GraphPassEntry = std::pair<GraphPass, GraphPassNameType>;

// Return currently registered passes. Passes are stored in a static vector
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPostPasses();
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType> >& getCustomPrePasses();
>>>>>>> 874e77e8ae... Make another attempt at PassManager for custom pass registration/deregistration.

TORCH_API GraphPassNameType registerPostPass(GraphPass p);
TORCH_API GraphPassNameType registerPrePass(GraphPass p);
using RegisterPass = RegisterPostPass;

// Look up pass by name passed in, remove it from registered passes
TORCH_API void ClearPostPass(GraphPassNameType p);
TORCH_API void ClearPrePass(GraphPassNameType p);

// Remove all passes
TORCH_API void ClearAllPostPasses();
TORCH_API void ClearAllPrePasses();

/*
 * PassManager is a wrapper on the register/clear PostPass functions above. It
 * will register the pass provided in "registerPass" and will hold on to its
 * associated name that way clearPass can be later called and will delete the
 * pass used to register when called.
 *
 * PassManager is templated because we want static variables based on a
 * particular GraphPass. When deriving from PassManager, you should send as the
 * template parameter your derived class as you would for the curiously
 * recurring template pattern. This template parameter isn't actually used and
 * is simply done to prevent static members from being shared across derived
 * types.
 */
template <typename DerivedType>
struct TORCH_API PassManager{
private:
  // We want this class to be abstract because it's 
  virtual void abstract() = 0;
protected:
 /*
  * isRegistered() will return if a pass has been registered
  * isRegistered(true) will change the value of the internal static bool
  * 
  * There's an internal static bool to this function to keep track of the state,
  * this is so when functions are derived from this class, they don't have to
  * worry about initializing the static members.
  */
 static bool isRegistered(bool flip_bit = false);
 /*
  * name() will return the name of the registered pass
  * name(pass_name, true) will set the name of the pass
  * Similarly to isRegistered we use an internal static variable to hold the name.
  */
 static GraphPassNameType name(
     GraphPassNameType PassName = 0,
     bool set = false);
public:
  // registerPass(pass) will register the pass provided and set the name/isRegistered functions appropriately
  static void registerPass(GraphPass p);
  // Calls ClearPostPass(name())
  static void clearPass();
};

} // namespace jit
} // namespace torch
