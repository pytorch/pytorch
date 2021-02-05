#include <torch/csrc/jit/backends/backend_detail.h>

#include <ATen/core/builtin_function.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/backends/backend_resolver.h>

namespace torch {
namespace jit {
namespace detail {
c10::FunctionSchema getPreprocessSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("mod", c10::AnyType::get());
  c10::Argument method_compile_spec(
      "method_compile_spec",
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get()));

  c10::FunctionSchema preprocessor_schema(
      "preprocess",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{mod});
  return preprocessor_schema;
}

c10::FunctionSchema getCompileSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("processed", c10::AnyType::get());
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  c10::Argument method_compile_spec("method_compile_spec", any_dict_ty);
  c10::Argument handles("handles", any_dict_ty);

  c10::FunctionSchema compile_schema(
      "compile",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{handles});
  return compile_schema;
}

c10::FunctionSchema getExecuteSchema() {
  auto any_list_ty = c10::ListType::create(c10::AnyType::get());
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument handle("handle", c10::AnyType::get());
  c10::Argument input("input", any_list_ty);
  c10::Argument output("output", any_list_ty);
  return c10::FunctionSchema(
      "execute",
      /*overload_name=*/"",
      /*arguments=*/{self, handle, input},
      /*returns=*/{output});
}

Module codegen_backend_module(const std::string& backend_name,
                              const Module& orig_module,
                              const c10::Dict<IValue, IValue>& method_compile_spec,
                              c10::DictTypePtr any_dict_ty) {
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

  // Generate LoweredModule.
  Module loweredModule(
      "torch.jit." + backend_name + "LoweredModule",
      std::make_shared<CompilationUnit>(),
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
      method_compile_spec,
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
    std::string method_name = e.key().toStringRef();
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

  // Run preprocess so that __processed_module is set correctly before
  // compilation.
  loweredModule.run_method(
      "__preprocess",
      cloned_module._ivalue(),
      method_compile_spec);

  // Call __setstate__ to ensure that the returned Module is ready to
  // run.
  auto state = at::ivalue::Tuple::create(
      method_compile_spec,
      loweredModule.attr("__processed_module"));
  loweredModule.run_method("__setstate__", state);
  return loweredModule;
}
} // namespace detail
} // namespace jit
} // namespace torch
