#include <memory>
#include <unordered_map>
#include <vector>

struct _frozen;

namespace torch {
namespace deploy {

struct builtin_registry_item {
  explicit builtin_registry_item(
      const char* _name,
      const struct _frozen* _frozen_modules);
  const char* name;
  const struct _frozen* frozen_modules;
  int num_modules;
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

 private:
  explicit builtin_registry() = default;
  std::unordered_map<std::string, int> name2idx_;
  std::vector<std::unique_ptr<builtin_registry_item>> items_;
};

class builtin_registerer {
 public:
  explicit builtin_registerer(
      const char* name,
      const struct _frozen* frozen_modules) {
    // note: don't call glog api in this method since this method is usually
    // called before glog get setup
    fprintf(
        stderr,
        "Registering torch::deploy builtin module %s (idx %lu)\n",
        name,
        builtin_registry::items().size());
    builtin_registry::register_builtin(
        std::make_unique<builtin_registry_item>(name, frozen_modules));
  }
};

} // namespace deploy
} // namespace torch

#define CONCAT_IMPL(s1, s2) s1##s2
#define CONCAT(s1, s2) CONCAT_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCAT(str, __LINE__)

#define REGISTER_TORCH_DEPLOY_BUILTIN(libname, frozen_modules) \
  static torch::deploy::builtin_registerer ANONYMOUS_VARIABLE( \
      builtin_registerer)(#libname, frozen_modules)
