#include <torch/csrc/jit/backends/backend_init.h>

#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_resolver.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {

// Get all types that are shared in the module hierarchy rooted at \p mod.
std::unordered_set<TypePtr> getSharedModuleTypes(Module& mod) {
  // Maintain a set of all TypePtrs.
  std::unordered_set<TypePtr> types;
  // Maintain another set of TypePtrs that have been encountered more than once.
  std::unordered_set<TypePtr> duplicate_types;

  // Iterate over all modules in the hierarchy, including the root.
  for (auto module : mod.modules()) {
    auto module_type = module.type();
    if (types.count(module_type) > 0) {
      duplicate_types.insert(module_type);
    }

    types.insert(module_type);
  }

  return duplicate_types;
}

// Selectively lower \p mod to a backend. \p to_backend
// is called to lower modules. \p modules_to_lower contains
// qualified names of submodules of \p mod that should be lowered.
void toBackendSelectiveImpl(
    Module& mod,
    const py::function& to_backend,
    const std::vector<std::string>& modules_to_lower,
    const std::unordered_set<TypePtr>& duplicate_types) {
  // This map will be used later to remap types in ancestor module graphs for
  // all lowered submodules.
  std::unordered_map<TypePtr, TypePtr> type_remap;

  // For each module that should be lowered:
  for (const auto& module_to_lower : modules_to_lower) {
    // Use QualifiedName to parse the qualified module names.
    c10::QualifiedName qual_module_name(module_to_lower);
    auto& atoms = qual_module_name.atoms();

    // Search through the module hierarchy using the atoms of
    // qual_module_name until current points to the module to
    // be lowered and parent points to its parent.
    Module current = mod;
    Module parent;

    for (size_t i = 0, e = atoms.size(); i < e; ++i) {
      IValue submodule = current.attr(atoms[i]);
      if (submodule.isModule()) {
        if (i == e - 1) {
          parent = current;
        }
        current = submodule.toModule();
      } else {
        std::stringstream err;
        err << "Attribute named " << atoms[i] << " is not a Module";
        throw std::runtime_error(err.str());
      }
    }

    // Check that the parent type is not shared and therefore can be edited.
    if (duplicate_types.count(parent.type()) > 0) {
      throw py::cast_error(c10::str(
          "Selective lowering is only supported for module hierarchies with unique types for selected modules; ",
          parent.type()->repr_str(),
          " is shared"));
    }

    // Call to_backend on the module that needs to be lowered. It needs to be
    // wrapped before doing so because _to_jit_backend accepts wrapped modules.
    // The result needs to be unwrapped in order to access its type below.
    auto lowered_submodule =
        py::cast<Module>(to_backend(py::module::import("torch.jit._recursive")
                                        .attr("wrap_cpp_module")(current))
                             .attr("_c"));

    // Adjust the parent's type so that the type of the submodule matches
    // the type of lowered_submodule.
    auto parent_type = parent.type();

    parent_type->unsafeChangeAttributeType(
        atoms.back(), lowered_submodule.type());
    parent.setattr(atoms.back(), lowered_submodule._ivalue());

    // Record the type mapping from old type -> lowered type.
    type_remap[current.type()] = lowered_submodule.type();
  }

  // Having lowered all of the modules that needed to be lowered, remap types in
  // all graphs in the hierarchy so that the graphs all use the new lowered
  // type.
  auto type_remap_fn = [&type_remap](TypePtr in) {
    auto it = type_remap.find(in);
    if (it == type_remap.end())
      return in;
    return it->second;
  };

  // modules() iterates over all modules in the hierarchy including the root.
  for (auto module : mod.modules()) {
    auto module_type = module.type();
    for (auto& fn : module_type->methods()) {
      auto method = module.get_method(fn->name());
      auto graph = method.graph();
      graph->remapTypes(type_remap_fn);
      auto new_schema = fn->getSchema().cloneWithRemappedTypes(type_remap_fn);
      fn->setSchema(new_schema);
    }
  }
}

Module codegen_func(
    const std::string& backend_name,
    const Module& orig_module,
    const py::dict& method_compile_spec) {
  // Represents of a Type of Dict[str, Any].
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  return detail::codegen_backend_module(
      backend_name,
      orig_module,
      toIValue(method_compile_spec, any_dict_ty).toGenericDict(),
      any_dict_ty);
}

void initJitBackendBindings(PyObject* module) {
  // Bind a function for lowering to each JIT backend. The name of the backend
  // must be the first argument. For example, to lower a Module to
  // "example_backend", declared as
  //
  //  static auto cls = torch::jit::backend<ExampleBackend>("example_backend");
  //
  // this function must be called like
  //
  //  torch._C._jit_to_backend("example_backend", module, spec)
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_jit_to_backend",
      [=](const std::string& backend_name,
          py::handle orig_module,
          const py::dict& method_compile_spec) {
        return py::module::import("torch.jit._recursive")
            .attr("wrap_cpp_module")(codegen_func(
                backend_name,
                py::cast<Module>(orig_module.attr("_c")),
                method_compile_spec));
      });

  m.def(
      "_jit_to_backend_selective",
      [=](py::handle orig_module,
          const py::function& to_backend,
          const std::vector<std::string>& modules_to_lower) {
        if (auto original_module =
                as_module(py::cast<py::object>(orig_module))) {
          // Clone the Module to avoid editing types that are shared with
          // Modules in other instances outside this hierarchy.
          Module& mod = original_module.value();
          auto cloned_mod = mod.clone();
          // Get all shared module types. Type sharing is only a problem if the
          // parent modules of the ones to lower are in this set.
          auto shared_types = getSharedModuleTypes(cloned_mod);
          toBackendSelectiveImpl(
              cloned_mod, to_backend, modules_to_lower, shared_types);
          // Wrap the result in a RecursiveScriptModule because that's what
          // the caller passed in.
          return py::module::import("torch.jit._recursive")
              .attr("wrap_cpp_module")(cloned_mod);
        }

        throw py::cast_error(c10::str(
            "Object ", py::str(orig_module), " is not a ScriptModule"));
      });
}
} // namespace jit
} // namespace torch
