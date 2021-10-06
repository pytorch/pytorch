#include <Python.h>
#include <c10/util/Exception.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

namespace torch {
namespace deploy {

// These numbers of modules should not change as long as the cpython version
// embedded in the build remains fixed
static const size_t NUM_FROZEN_PY_BUILTIN_MODULES = 6;
static const size_t NUM_FROZEN_PY_STDLIB_MODULES = 680;

extern "C" struct _frozen _PyImport_FrozenModules[];
extern "C" struct _frozen _PyImport_FrozenModules_torch[];
REGISTER_TORCH_DEPLOY_BUILTIN(cpython_internal, PyImport_FrozenModules);
REGISTER_TORCH_DEPLOY_BUILTIN(frozenpython, _PyImport_FrozenModules);
REGISTER_TORCH_DEPLOY_BUILTIN(frozentorch, _PyImport_FrozenModules_torch);

BuiltinRegistryItem::BuiltinRegistryItem(
    const char* _name,
    const struct _frozen* _frozenModules)
    : name(_name), frozenModules(_frozenModules) {
  numModules = 0;
  if (frozenModules) {
    while (frozenModules[numModules].name != nullptr) {
      ++numModules;
    }
  }

  fprintf(
      stderr,
      "torch::deploy builtin %s contains %d modules\n",
      name,
      numModules);
}

BuiltinRegistry* BuiltinRegistry::get() {
  static BuiltinRegistry _registry;
  return &_registry;
}

void BuiltinRegistry::registerBuiltin(
    std::unique_ptr<BuiltinRegistryItem> item) {
  if (get()->name2idx_.find(item->name) != get()->name2idx_.end()) {
    throw std::runtime_error(std::string("redefine bultin: ") + item->name);
  }
  get()->name2idx_[item->name] = get()->items_.size();
  get()->items_.emplace_back(std::move(item));
}

BuiltinRegistryItem* BuiltinRegistry::getItem(const std::string& name) {
  auto itr = get()->name2idx_.find(name);
  return itr == get()->name2idx_.end() ? nullptr
                                       : get()->items_[itr->second].get();
}

int BuiltinRegistry::totalNumModules() {
  int tot = 0;
  for (const auto& itemptr : get()->items_) {
    tot += itemptr->numModules;
  }
  return tot;
}

struct _frozen* BuiltinRegistry::getAllFrozenModules() {
  /* Allocate new memory for the combined table */
  int totNumModules = totalNumModules();
  struct _frozen* p = nullptr;
  if (totNumModules > 0 &&
      totNumModules <= SIZE_MAX / sizeof(struct _frozen) - 1) {
    size_t size = sizeof(struct _frozen) * (totNumModules + 1);
    p = (_frozen*)PyMem_Malloc(size);
  }
  if (p == nullptr) {
    return nullptr;
  }

  // mark p as an empty frozen module list
  memset(&p[0], 0, sizeof(p[0]));

  /* Copy the tables into the new memory */
  int off = 0;
  for (const auto& itemptr : items()) {
    if (itemptr->numModules > 0) {
      memcpy(
          p + off,
          itemptr->frozenModules,
          (itemptr->numModules + 1) * sizeof(struct _frozen));
      off += itemptr->numModules;
    }
  }

  return p;
}

void BuiltinRegistry::sanityCheck() {
  auto* cpythonInternalFrozens = getItem("cpython_internal");
  // Num frozen builtins shouldn't change (unless modifying the underlying
  // cpython version)
  TORCH_INTERNAL_ASSERT(
      cpythonInternalFrozens != nullptr &&
          cpythonInternalFrozens->numModules == NUM_FROZEN_PY_BUILTIN_MODULES,
      "Missing python builtin frozen modules");

  auto* frozenpython = getItem("frozenpython");
  auto* frozentorch = getItem("frozentorch");
  // Check frozenpython+frozentorch together since in OSS frozenpython is empty
  // and frozentorch contains stdlib+torch, while in fbcode they are separated
  // due to thirdparty2 frozenpython. No fixed number of torch modules to check
  // for, but there should be at least one.
  TORCH_INTERNAL_ASSERT(
      frozenpython != nullptr && frozentorch != nullptr &&
          frozenpython->numModules + frozentorch->numModules >
              NUM_FROZEN_PY_STDLIB_MODULES + 1,
      "Missing frozen python stdlib or torch modules");
}

} // namespace deploy
} // namespace torch
