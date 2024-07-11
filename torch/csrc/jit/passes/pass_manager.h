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

// Since Passes are std::functions, we associate a UUID to each pass, this way
// if we want to deregister a pass, we have something to reference it by.
using GraphPassNameType = unsigned int;

// Graph pass entries have a name associated with them
using GraphPassEntry = std::pair<GraphPass, GraphPassNameType>;

// Return currently registered passes. Passes are stored in a static vector
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPostPasses();
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPrePasses();

TORCH_API GraphPassNameType registerPostPass(GraphPass p);
TORCH_API GraphPassNameType registerPrePass(GraphPass p);

// Look up pass by name passed in, remove it from registered passes
TORCH_API void clearPostPass(GraphPassNameType p);
TORCH_API void clearPrePass(GraphPassNameType p);

// Remove all passes
TORCH_API void clearAllPostPasses();
TORCH_API void clearAllPrePasses();

// LEGACY CALL
struct TORCH_API RegisterPostPass {
  RegisterPostPass(GraphPass p);
};

using RegisterPass = RegisterPostPass;

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
struct C10_EXPORT PassManager {
 private:
  // We want this class to be abstract because it's
  virtual void abstract() = 0;

 protected:
  /*
   * isRegistered() will return if a pass has been registered
   * isRegistered(true) will change the value of the internal static bool
   *
   * There's an internal static bool to this function to keep track of the
   * state, this is so when functions are derived from this class, they don't
   * have to worry about initializing the static members.
   */
  static bool isRegistered(bool flip_bit = false) {
    static bool val = false;
    if (flip_bit)
      val = !val;
    return val;
  }

  /*
   * name() will return the name of the registered pass
   * name(pass_name, true) will set the name of the pass
   * Similarly to isRegistered we use an internal static variable to hold the
   * name.
   */
  static GraphPassNameType passID(
      GraphPassNameType PassID = 0,
      bool set = false) {
    static GraphPassNameType pass_id = 0;
    if (set)
      pass_id = PassID;
    return pass_id;
  }

 public:
  // registerPass(pass) will register the pass provided and set the
  // name/isRegistered functions appropriately, it returns a bool value
  // indicating whether the given pass is already registered previously.
  static bool registerPass(GraphPass p) {
    if (!isRegistered()) {
      // If we don't already have a registered pass, register pass
      // hold on to its name, change isRegistered to true
      passID(registerPostPass(std::move(p)), true);
      isRegistered(true);
      return false;
    }
    return true;
  }

  // Calls ClearPostPass(passID())
  static void clearPass() {
    // If the pass is registered, clear it and change isRegistered to false.
    if (isRegistered()) {
      clearPostPass(passID());
      isRegistered(true);
    }
  }

  // clang-tidy requires virtual destructor;
  virtual ~PassManager() = default;
};

} // namespace jit
} // namespace torch
