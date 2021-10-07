/*
 * The torch::deploy builtin registry library is used to make adding new bultins
 * to torch::deploy easy and clean.
 *
 * Under the hood, to add a torch::deploy builtin, the following things need to
 * be done
 * 1. merge the frozen modules for the builtin into PyImport_FrozenModules
 * 2. appending PyInit methods for modules implemented in C++ to the CPython
 *    builtin module list via methods like PyImport_AppendInittab
 * 3. tweak the sys.meta_path a bit to force loading non-toplevel moduels for the
 *    torch::deploy builtin via the CPython builtin module importer.
 *
 * Doing all these things again and again manually is cumbersome and error-prone.
 * This builtin registry library supports open registration for torch::deploy
 * builtins. It does the work above by a single line of code invoking
 * REGISTER_TORCH_DEPLOY_BUILTIN macro. Here is an example for numpy:
 *
 *   REGISTER_TORCH_DEPLOY_BUILTIN(numpy, numpy_frozen_modules, <list of name, PyInit function pairs>)
 *
 * Calling REGISTER_TORCH_DEPLOY_BUILTIN macro will instantiate a BuiltinRegisterer
 * object. The constructor of BuiltinRegisterer does the real registration work.
 */
#include <memory>
#include <unordered_map>
#include <vector>

struct _frozen;

namespace torch {
namespace deploy {

/*
 * This data structure describes a torch::deploy builtin being registered to
 * the registry.
 *
 * Each torch::deploy builtin contains the following basically information:
 * - a name for the builtin. It's usually the name of the library like numpy
 * - the lsit of frozen modules
 * - the list of builtin modules
 */
struct BuiltinRegistryItem {
  explicit BuiltinRegistryItem(
      const char* _name,
      const struct _frozen* _frozenModules);
  const char* name;
  const struct _frozen* frozenModules;
  int numModules;
};

/*
 * BuiltinRegistry maintains all the registered torch::deploy builtins. This
 * class is a singleton. Calling BuiltinRegistry::get() returns the single object
 * instance.
 *
 * The state of this class is basically a list of BuiltinRegistryItem registered
 * so far.
 */
class BuiltinRegistry {
 public:
  static BuiltinRegistry* get();
  // for unittest
  static void clear() {
    get()->items_.clear();
  }
  static void registerBuiltin(std::unique_ptr<BuiltinRegistryItem> item);
  static const std::vector<std::unique_ptr<BuiltinRegistryItem>>& items() {
    return get()->items_;
  }
  static BuiltinRegistryItem* getItem(const std::string& name);
  static int totalNumModules();
  static struct _frozen* getAllFrozenModules();
  // call this after all the registration is done.
  static void sanityCheck();

 private:
  explicit BuiltinRegistry() = default;
  std::unordered_map<std::string, int> name2idx_;
  std::vector<std::unique_ptr<BuiltinRegistryItem>> items_;
};

/*
 * If nobody defines allowLibrary method, allowLibrary will be evaluated to
 * 0 and we allow registering any libraries. If someone defines allowLibrary,
 * we respect that and only registering libraries that get true from calling
 * allowLibrary(libname).
 *
 * Currently used in unit test so we can fully control the registered libraries.
 */
__attribute__((weak)) bool allowLibrary(const std::string& libname);

/*
 * This class implements RAII (resource acquisition is initialization) to
 * register a bulitin to the registry.
 */
class BuiltinRegisterer {
 public:
  explicit BuiltinRegisterer(
      const char* name,
      const struct _frozen* frozenModules) {
    if (allowLibrary && !allowLibrary(name)) {
      fprintf(stderr, "Skip %s since it's rejected by the allowLibrary method\n", name);
      return;
    }
    // note: don't call glog api in this method since this method is usually
    // called before glog get setup
    fprintf(
        stderr,
        "Registering torch::deploy builtin module %s (idx %lu)\n",
        name,
        BuiltinRegistry::items().size());
    BuiltinRegistry::registerBuiltin(
        std::make_unique<BuiltinRegistryItem>(name, frozenModules));
  }
};

} // namespace deploy
} // namespace torch

#define CONCAT_IMPL(s1, s2) s1##s2
#define CONCAT(s1, s2) CONCAT_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCAT(str, __LINE__)

#define REGISTER_TORCH_DEPLOY_BUILTIN(libname, frozenModules) \
  static torch::deploy::BuiltinRegisterer ANONYMOUS_VARIABLE( \
      BuiltinRegisterer)(#libname, frozenModules)
