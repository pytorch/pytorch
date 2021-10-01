#include <cstdarg>
#include <memory>
#include <unordered_map>
#include <vector>

struct _frozen;

namespace torch {
namespace deploy {

struct builtin_registry_item {
  explicit builtin_registry_item(
      const char* _name,
      const struct _frozen* _frozen_modules,
      std::vector<std::pair<const char*, void*>>&& _builtin_modules);
  const char* name;
  const struct _frozen* frozen_modules;
  int num_modules;
  std::vector<std::pair<const char*, void*>> builtin_modules;
};

class builtin_registry {
 public:
  static builtin_registry* get();
  // for unittest
  static void clear() {
    get()->items_.clear();
  }
  static void register_builtin(std::unique_ptr<builtin_registry_item> item);
  static const std::vector<std::unique_ptr<builtin_registry_item>>& items() {
    return get()->items_;
  }
  static builtin_registry_item* get_item(const std::string& name);
  static int total_num_modules();
  static struct _frozen* get_all_frozen_modules();
  static std::vector<std::pair<const char*, void*>> get_all_builtin_modules();
  static void append_cpython_inittab();
  static std::string get_builtin_modules_csv();

 private:
  explicit builtin_registry() = default;
  std::unordered_map<std::string, int> name2idx_;
  std::vector<std::unique_ptr<builtin_registry_item>> items_;
};

class builtin_registerer {
 public:
  explicit builtin_registerer(
      const char* name,
      const struct _frozen* frozen_modules...) {
    // gather builtin modules for this lib
    va_list args;
    va_start(args, frozen_modules);
    const char* module_name = nullptr;
    void* init_fn = nullptr;
    std::vector<std::pair<const char*, void*>> builtin_modules;
    while (true) {
      module_name = va_arg(args, const char*);
      // encounter end of sequence
      if (module_name == nullptr) {
        break;
      }
      init_fn = va_arg(args, void*);
      // skip null init function. This can happen if we create weak reference
      // to init functions defined in another library. Depending on if we
      // link with that library, the init function pointer will be the real
      // implementation or nullptr. tensorrt is a good example. If this is
      // a CPU build, we will not link with the tensorrt library, so the init
      // function will be nullptr; on the other hand if this is a GPU build,
      // we link with the tensorrt library, so the init function will not be
      // nullptr.
      if (init_fn == nullptr) {
        continue;
      }
      builtin_modules.emplace_back(std::make_pair(module_name, init_fn));
    }

    // note: don't call glog api in this method since this method is usually
    // called before glog get setup
    fprintf(
        stderr,
        "Registering torch::deploy builtin library %s (idx %lu) with %lu builtin modules\n",
        name,
        builtin_registry::items().size(),
        builtin_modules.size());

    builtin_registry::register_builtin(std::make_unique<builtin_registry_item>(
        name, frozen_modules, std::move(builtin_modules)));
  }
};

} // namespace deploy
} // namespace torch

#define CONCAT_IMPL(s1, s2) s1##s2
#define CONCAT(s1, s2) CONCAT_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCAT(str, __LINE__)

/* there can be a variable list of builtin modules following frozen_modules
 * A typical usage of this macro is:
 *
 *  REGISTER_TORCH_DEPLOY_BUILTIN(library_name_without_quote,
 * frozen_modules_list, builtin_module_name_1, builtin_module_init_function_1,
 * ..., builtin_module_name_N, builtin_module_init_function_N)
 */
#define REGISTER_TORCH_DEPLOY_BUILTIN(libname, frozen_modules...) \
  static torch::deploy::builtin_registerer ANONYMOUS_VARIABLE(    \
      builtin_registerer)(#libname, frozen_modules, nullptr)
