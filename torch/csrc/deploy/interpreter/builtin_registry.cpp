#include <Python.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

namespace torch {
namespace deploy {

// These numbers of modules should not change as long as the cpython version
// embedded in the build remains fixed
static const size_t NUM_FROZEN_PY_BUILTIN_MODULES = 6;
static const size_t NUM_FROZEN_PY_STDLIB_MODULES = 680;

extern "C" struct _frozen _PyImport_FrozenModules[];
extern "C" struct _frozen _PyImport_FrozenModules_torch[];
extern "C" PyObject* initModule(void);
REGISTER_TORCH_DEPLOY_BUILTIN(cpython_internal, PyImport_FrozenModules);
REGISTER_TORCH_DEPLOY_BUILTIN(frozenpython, _PyImport_FrozenModules);
REGISTER_TORCH_DEPLOY_BUILTIN(
    frozentorch,
    _PyImport_FrozenModules_torch,
    "torch._C",
    initModule);

BuiltinRegistryItem::BuiltinRegistryItem(
    const char* _name,
    const struct _frozen* _frozenModules,
    std::vector<std::pair<const char*, void*>>&& _builtinModules)
    : name(_name),
      frozenModules(_frozenModules),
      builtinModules(std::move(_builtinModules)) {
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

void BuiltinRegistry::runPreInitialization() {
  TORCH_INTERNAL_ASSERT(!Py_IsInitialized());
  sanityCheck();

  PyImport_FrozenModules = BuiltinRegistry::getAllFrozenModules();
  TORCH_INTERNAL_ASSERT(PyImport_FrozenModules != nullptr);

  appendCPythonInittab();
}

const char* metaPathSetupTemplate = R"PYTHON(
import sys
# We need to register a custom meta path finder because we are registering
# `torch._C` as a builtin module.
#
# Normally, builtins will be found by the `BuiltinImporter` meta path finder.
# However, `BuiltinImporter` is hard-coded to assume that all builtin modules
# are top-level imports.  Since `torch._C` is a submodule of `torch`, the
# BuiltinImporter skips it.
class F:
    def find_spec(self, fullname, path, target=None):
        if fullname in [<<<DEPLOY_BUILTIN_MODULES_CSV>>>]:
            # Load this module using `BuiltinImporter`, but set `path` to None
            # in order to trick it into loading our module.
            return sys.meta_path[1].find_spec(fullname, path=None, target=None)
        return None
sys.meta_path.insert(0, F())
)PYTHON";

void BuiltinRegistry::runPostInitialization() {
  TORCH_INTERNAL_ASSERT(Py_IsInitialized());
  std::string metaPathSetupScript(metaPathSetupTemplate);
  std::string replaceKey = "<<<DEPLOY_BUILTIN_MODULES_CSV>>>";
  auto itr = metaPathSetupScript.find(replaceKey);
  if (itr != std::string::npos) {
    metaPathSetupScript.replace(itr, replaceKey.size(), getBuiltinModulesCSV());
  }
  int r = PyRun_SimpleString(metaPathSetupScript.c_str());
  TORCH_INTERNAL_ASSERT(r == 0);
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

std::vector<std::pair<const char*, void*>> BuiltinRegistry::
    getAllBuiltinModules() {
  std::vector<std::pair<const char*, void*>> allBuiltinModules;
  for (const auto& itemptr : items()) {
    allBuiltinModules.insert(
        allBuiltinModules.end(),
        itemptr->builtinModules.begin(),
        itemptr->builtinModules.end());
  }
  return allBuiltinModules;
}

void BuiltinRegistry::appendCPythonInittab() {
  for (const auto& pair : get()->getAllBuiltinModules()) {
    PyImport_AppendInittab(
        pair.first, reinterpret_cast<PyObject* (*)()>(pair.second));
  }
}

std::string BuiltinRegistry::getBuiltinModulesCSV() {
  std::string modulesCSV;
  for (const auto& pair : get()->getAllBuiltinModules()) {
    if (!modulesCSV.empty()) {
      modulesCSV += ", ";
    }
    modulesCSV += fmt::format("'{}'", pair.first);
  }
  return modulesCSV;
}

BuiltinRegisterer::BuiltinRegisterer(
    const char* name,
    const struct _frozen* frozenModules...) {
  if (allowLibrary && !allowLibrary(name)) {
    fprintf(
        stderr,
        "Skip %s since it's rejected by the allowLibrary method\n",
        name);
    return;
  }
  // gather builtin modules for this lib
  va_list args;
  va_start(args, frozenModules);
  const char* moduleName = nullptr;
  void* initFn = nullptr;
  std::vector<std::pair<const char*, void*>> builtinModules;
  while (true) {
    moduleName = va_arg(args, const char*);
    // encounter end of sequence
    if (moduleName == nullptr) {
      break;
    }
    initFn = va_arg(args, void*);
    // skip null init function. This can happen if we create weak reference
    // to init functions defined in another library. Depending on if we
    // link with that library, the init function pointer will be the real
    // implementation or nullptr. tensorrt is a good example. If this is
    // a CPU build, we will not link with the tensorrt library, so the init
    // function will be nullptr; on the other hand if this is a GPU build,
    // we link with the tensorrt library, so the init function will not be
    // nullptr.
    if (initFn == nullptr) {
      continue;
    }
    builtinModules.emplace_back(moduleName, initFn);
  }

  // note: don't call glog api in this method since this method is usually
  // called before glog get setup
  fprintf(
      stderr,
      "Registering torch::deploy builtin library %s (idx %lu) with %lu builtin modules\n",
      name,
      BuiltinRegistry::items().size(),
      builtinModules.size());
  BuiltinRegistry::registerBuiltin(std::make_unique<BuiltinRegistryItem>(
      name, frozenModules, std::move(builtinModules)));
}

} // namespace deploy
} // namespace torch
