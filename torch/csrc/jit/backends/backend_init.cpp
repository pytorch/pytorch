#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_resolver.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {

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
    const c10::QualifiedName qual_backend_name({"__torch__",
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
    // This is the original cloned and preprocessed module.
    loweredModule.register_attribute(
        "__processed_module",
        AnyType::get(),
        cloned_module._ivalue(),
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

    // This loop generates one method on the LoweredModule for every key
    // in method_compile_spec.
    for (auto& e : method_compile_spec) {
      std::string method_name = py::cast<std::string>(e.first);
      static const auto method_ct = CodeTemplate(R"(
            def $method(self${,def_inputs}):
                typed_inputs: List[Any] = [${fwd_inputs,}]
                $ret, = self.__backend.execute(self.__handles["$method"], typed_inputs)
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

      if (auto out_tuple_ty = out_ty->cast<TupleType>()) {
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
      method_te.s("ret", out_ss.str());

      loweredModule.define(
          method_ct.format(method_te), loweredModuleResolver());
    }

    // Run preprocess so that __processed_module is set correctly before
    // compilation.
    loweredModule.run_method(
        "__preprocess",
        cloned_module._ivalue(),
        toIValue(method_compile_spec, any_dict_ty).toGenericDict());

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
          const Module& orig_module,
          const py::dict& method_compile_spec) {
        return py::module::import("torch.jit._recursive")
            .attr("wrap_cpp_module")(
                codegen_lambda(backend_name, orig_module, method_compile_spec));
      });
}
} // namespace jit
} // namespace torch
