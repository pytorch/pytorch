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
  auto codegen_lambda = [=](const std::string& backend_name,
                            const Module& orig_module,
                            const py::dict& method_compile_spec) {
    const c10::QualifiedName qual_backend_name(
        {"__torch__",
         "torch",
         "classes",
         detail::kBackendsNamespace,
         backend_name});
    // TODO: Validate method_compile_spec.

    // Clone orig_module to make sure backend transformation is
    // functional.
    auto cloned_module = orig_module.clone();

    // Represents of a Type of Dict[str, Any].
    auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

    // Generate LoweredModule.
    Module loweredModule(
        "torch.jit." + backend_name + "LoweredModule",
        get_python_cu(),
        /*shouldMangle=*/true);

    // Generate attributes.
    // This is the preprocessed module.
    // For backwards compatibility, for backends that implement preprocessing in
    // the backend interface rather than as a separate function, we just pass
    // the cloned original Module.
    loweredModule.register_attribute(
        "__processed_module",
        AnyType::get(),
        detail::hasBackendPreprocessFunction(backend_name)
            ? detail::getBackendPreprocessFunction(backend_name)(
                  cloned_module,
                  toIValue(method_compile_spec, any_dict_ty).toGenericDict())
            : cloned_module._ivalue(),
        /*is_param=*/false);

    // This is for the method_compile_spec passed in to to_<backend> or
    // loaded from an exported model.
    loweredModule.register_attribute(
        "__method_compile_spec",
        any_dict_ty,
        toIValue(method_compile_spec, any_dict_ty).toGenericDict(),
        /*is_param=*/false);

    // This is a pointer to a backend instance that is used to access
    // compile and execute functions.
    auto cls = getCustomClass(qual_backend_name.qualifiedName());
    TORCH_INTERNAL_ASSERT(cls);
    c10::intrusive_ptr<torch::CustomClassHolder> backend;
    loweredModule.register_attribute(
        "__backend", cls, IValue::make_capsule(backend));

    // This is the list of opaque backend handles returned by
    // backend.compile.
    loweredModule.register_attribute(
        "__handles",
        any_dict_ty,
        c10::impl::GenericDict(
            any_dict_ty->getKeyType(), any_dict_ty->getValueType()),
        /*is_param=*/false);

    // Methods.

    // This is a helper function for creating a new instance of the
    // backend class.
    static const auto create_backend_ct = CodeTemplate(R"(
              def __create_backend(self):
                  self.__backend = $name()
              )");
    TemplateEnv create_backend_te;
    create_backend_te.s("name", qual_backend_name.qualifiedName());
    loweredModule.define(
        create_backend_ct.format(create_backend_te), loweredModuleResolver());

    // getstate and setstate are for serialization/deserialization of
    // the LoweredModule.
    loweredModule.define(
        R"(
              def __getstate__(self):
                  return self.__method_compile_spec, self.__processed_module
              )",
        loweredModuleResolver());

    loweredModule.define(
        R"(
              def __setstate__(self, state):
                  self.__method_compile_spec = state[0]
                  self.__processed_module = state[1]
                  self.__create_backend()
                  self.__handles = self.__backend.compile(self.__processed_module, self.__method_compile_spec)
              )",
        loweredModuleResolver());

    // Only add preprocess method to the LoweredModule if there is no
    // standalone BackendPreprocessFunction for this backend.
    // Kept for backwards compatibility for backends that implement
    // preprocessing in the backend interface rather than as a separate
    // function.
    if (!detail::hasBackendPreprocessFunction(backend_name)) {
      // This is never called during compilation or execution, but is
      // needed to generate the LoweredModule because we don't have access
      // to an instance of the backend as a C++ object with which to call
      // preprocess.
      loweredModule.define(
          R"(
                def __preprocess(self, mod: Any, method_compile_spec: Dict[str, Any]):
                    self.__create_backend()
                    self.__processed_module = self.__backend.preprocess(mod, method_compile_spec)
              )",
          loweredModuleResolver());
      // Run preprocess so that __processed_module is set correctly before
      // compilation.
      loweredModule.run_method(
          "__preprocess",
          cloned_module._ivalue(),
          toIValue(method_compile_spec, any_dict_ty).toGenericDict());
    }

    // This loop generates one method on the LoweredModule for every key
    // in method_compile_spec.
    for (auto& e : method_compile_spec) {
      std::string method_name = py::cast<std::string>(e.first);
      static const auto method_ct = CodeTemplate(R"(
              def $method(self${,def_inputs}):
                  typed_inputs: List[Any] = [${fwd_inputs,}]
                  $unpack, = self.__backend.execute(self.__handles["$method"], typed_inputs)
                  ${refine,}
                  return $ret
              )");

      TemplateEnv method_te;
      method_te.s("method", method_name);
      auto method = orig_module.get_method(method_name);
      auto& function = method.function();
      auto& schema = function.getSchema();

      // Generate the inputs for the function signature (def_inputs) and
      // for passing to backend.execute (fwd_inputs).
      std::vector<std::string> def_inputs, fwd_inputs;
      for (const auto& arg : schema.arguments()) {
        auto name = arg.name();

        // Skip self since that is only and always present in the
        // signature.
        if (name == "self") {
          continue;
        }

        auto default_value = arg.default_value();

        if (arg.kwarg_only()) {
          // If this is a kwarg, it needs to be emitted as keyword=value
          // in the definition and keyword=keyword in the call to
          // backend_execute.
          TORCH_INTERNAL_ASSERT(default_value.has_value());
          std::stringstream def_ss, fwd_ss;
          def_ss << name << "=";
          fwd_ss << name << "=" << name;
          default_value->repr(def_ss, [](std::ostream&, const IValue&) -> bool {
            return false;
          });
          def_inputs.emplace_back(def_ss.str());
          fwd_inputs.emplace_back(fwd_ss.str());
        } else {
          // If this is not a kwarg, it should be emitted as is in the
          // signature and the call to backend_execute.
          def_inputs.emplace_back(name);
          fwd_inputs.emplace_back(name);
        }
      }

      // Generate a comma-delimited list of identifiers to unpack
      // outputs, as well as a list of isinstance checks to make sure
      // the backend returned the types it was supposed to.
      std::stringstream out_ss, type_check_ss;
      std::vector<std::string> type_checks;
      TORCH_INTERNAL_ASSERT(schema.returns().size() == 1);
      auto out_ty = schema.returns().at(0).type();

      out_ss << "_0";
      type_check_ss << "assert isinstance(_0, ";

      auto out_tuple_ty = out_ty->cast<TupleType>();

      if (out_tuple_ty) {
        auto tuple_elements = out_tuple_ty->elements();
        type_check_ss << tuple_elements[0]->str() << ")";
        type_checks.emplace_back(type_check_ss.str());
        for (unsigned i = 1, e = tuple_elements.size(); i < e; ++i) {
          type_check_ss.str(std::string());
          type_check_ss.clear();
          out_ss << ", _" << i;
          type_check_ss << "assert isinstance(_" << i << ", "
                        << tuple_elements[i]->str() << ")";
          type_checks.emplace_back(type_check_ss.str());
        }
      } else {
        type_check_ss << out_ty->str() << ")";
        type_checks.emplace_back(type_check_ss.str());
      }

      method_te.v("def_inputs", def_inputs);
      method_te.v("fwd_inputs", fwd_inputs);
      method_te.v("refine", type_checks);
      method_te.s("unpack", out_ss.str());

      // If the output type is a single element tuple then add an extra comma
      // to ensure the final output maintains this type.
      if (out_tuple_ty && out_tuple_ty->elements().size() == 1) {
        out_ss << ",";
      }

      method_te.s("ret", out_ss.str());

      loweredModule.define(
          method_ct.format(method_te), loweredModuleResolver());
    }

    // Call __setstate__ to ensure that the returned Module is ready to
    // run.
    auto state = at::ivalue::Tuple::create(
        toIValue(method_compile_spec, any_dict_ty).toGenericDict(),
        loweredModule.attr("__processed_module"));
    loweredModule.run_method("__setstate__", state);
    return loweredModule;
  };
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_jit_to_backend",
      [=](const std::string& backend_name,
          py::handle orig_module,
          const py::dict& method_compile_spec) {
        return py::module::import("torch.jit._recursive")
            .attr("wrap_cpp_module")(codegen_lambda(
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
