#include <torch/csrc/deploy/builtin_registry.h>
#include <Python.h>

namespace torch {
namespace deploy {

builtin_registry_item::builtin_registry_item(
    const char* _name,
    const struct _frozen* _frozen_modules)
    : name(_name), frozen_modules(_frozen_modules) {
  num_modules = 0;
  if (frozen_modules) {
    while (frozen_modules[num_modules].name != nullptr) {
      ++num_modules;
    }
  }

  fprintf(
      stderr,
      "torch::deploy builtin %s contains %d modules\n",
      name,
      num_modules);
}

builtin_registry* builtin_registry::get() {
  static builtin_registry _registry;
  return &_registry;
}

void builtin_registry::register_builtin(
    std::unique_ptr<builtin_registry_item> item) {
  if (get()->name2idx_.find(item->name) != get()->name2idx_.end()) {
    throw std::runtime_error(std::string("redefine bultin: ") + item->name);
  }
  get()->name2idx_[item->name] = get()->items_.size();
  get()->items_.emplace_back(std::move(item));
}

builtin_registry_item* builtin_registry::get_item(const std::string& name) {
  auto itr = get()->name2idx_.find(name);
  return itr == get()->name2idx_.end() ? nullptr
                                       : get()->items_[itr->second].get();
}

int builtin_registry::total_num_modules() {
  int tot = 0;
  for (const auto& itemptr : get()->items_) {
    tot += itemptr->num_modules;
  }
  return tot;
}

struct _frozen* builtin_registry::get_all_frozen_modules() {
  /* Allocate new memory for the combined table */
  int tot_num_modules = total_num_modules();
  struct _frozen* p = nullptr;
  if (tot_num_modules > 0 &&
      tot_num_modules <= SIZE_MAX / sizeof(struct _frozen) - 1) {
    size_t size = sizeof(struct _frozen) * (tot_num_modules + 1);
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
    if (itemptr->num_modules > 0) {
      memcpy(
          p + off,
          itemptr->frozen_modules,
          (itemptr->num_modules + 1) * sizeof(struct _frozen));
      off += itemptr->num_modules;
    }
  }

  return p;
}

} // namespace deploy
} // namespace torch
