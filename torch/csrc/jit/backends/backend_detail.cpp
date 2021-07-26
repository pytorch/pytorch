#include <torch/csrc/jit/backends/backend_detail.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/backends/backend_resolver.h>
#include <torch/csrc/jit/frontend/code_template.h>

#include <memory>
#include <stack>
#include <unordered_map>

namespace torch {
namespace jit {
namespace detail {
namespace {

/*
 * This is the API via which backend's preprocess function will obtain debug
 * handles corresponding to the nodes of the graph for the lowered methods of
 * the module.
 * Implementation: Given graph
 * For each node of the graph, request debug handle via debug_info_recorder.
 * debug_info_recorder returns the next debug handle and record node with
 * corresponding debug info, such as source range and inlined callstack.
 *
 * Backend code for lowering module, preprocess, calls
 * generate_debug_handles(graph)) which will return debug handles corresponding
 * to the Node* of the said graph.
 *
 * In to_backend, after lowering, stopRecording is called on
 * BackendModuleDebugInfoRecorder: It will extract debug map. This map gets
 * stored as part of the lowered module.
 * During serialization, specifically for bytecode serialization, check is made
 * to see if the model being serialized has any lowered modules. If so
 * corresponding debug map is extracted and serialized.
 */

NodeToDebugHandle generate_debug_handles(
    BackendDebugInfoRecorder& debug_info_recorder,
    const std::shared_ptr<Graph>& graph) {
  NodeToDebugHandle node_to_debug_handles;

  std::stack<Block*> blocks_to_visit;
  // TODO: Look into using DepthFirstGraphNodeIterator
  // At the moment it takes non-const graph but maybe we can make it
  // general such that it can work with both.
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      DebugHandleType debug_handle = debug_info_recorder.getNextDebugHandle(n);
      node_to_debug_handles.emplace(n, debug_handle);
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return node_to_debug_handles;
}

std::unordered_map<std::string, BackendPreprocessFunction>&
backendPreprocessFunctions() {
  static std::unordered_map<std::string, BackendPreprocessFunction>
      preprocess_functions;
  return preprocess_functions;
}
} // namespace

bool hasBackendPreprocessFunction(const std::string& name) {
  return backendPreprocessFunctions().count(name);
}

void registerBackendPreprocessFunction(
    const std::string& name,
    const BackendPreprocessFunction& preprocess) {
  TORCH_CHECK(
      !detail::hasBackendPreprocessFunction(name),
      "Preprocessing function for backend ",
      name,
      " is already registered. Ensure that registration is only called once.");
  detail::backendPreprocessFunctions()[name] = preprocess;
}

BackendPreprocessFunction getBackendPreprocessFunction(
    const std::string& name) {
  TORCH_CHECK(
      hasBackendPreprocessFunction(name),
      "Preprocessing function for backend ",
      name,
      " is not registered.");
  return backendPreprocessFunctions()[name];
}

Module codegen_backend_module(
    const std::string& backend_name,
    const Module& orig_module,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const c10::DictTypePtr& any_dict_ty) {
  const c10::QualifiedName qual_backend_name(
      {"__torch__", "torch", "classes", kBackendsNamespace, backend_name});
  // TODO: Validate method_compile_spec.

  // Clone orig_module to make sure backend transformation is
  // functional.
  auto cloned_module = orig_module.clone();
  auto module_name = orig_module.type()->name()->qualifiedName();
  // Generate LoweredModule.
  Module loweredModule(
      "torch.jit.LoweredModule." + backend_name + "." + module_name,
      std::make_shared<CompilationUnit>(),
      /*shouldMangle=*/true);

  // 1. Initialized debug info recorder.
  // 2. Later call debug_info_recorder.stopRecording() to gather
  //    recorded debug info and save it in __backend_debug_info.
  BackendDebugInfoRecorder debug_info_recorder;

  // Generate attributes.
  // This is the preprocessed module.
  // For backwards compatibility, for backends that implement preprocessing in
  // the backend interface rather than as a separate function, we just pass
  // the cloned original Module.

  BackendDebugHandleGenerator debug_handle_generator =
      [&](const std::shared_ptr<Graph>& g) {
        return generate_debug_handles(debug_info_recorder, g);
      };
  loweredModule.register_attribute(
      "__processed_module",
      AnyType::get(),
      detail::getBackendPreprocessFunction(backend_name)(
          cloned_module, method_compile_spec, debug_handle_generator),
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

  // Helper function to expose backend.is_available() to Module generation code.
  // Assumes self.__backend exists (i.e. __create_backend() has already been
  // invoked).
  loweredModule.define(
      R"(
            def __is_available(self):
                return self.__backend.is_available()
            )",
      loweredModuleResolver());

  // backend_debug_info_class is an instance of BackendDebugInfo that
  // stores debug information.
  // The purpose of this class is to make the debug information available
  // at model saving time for serializing it outside of the lowered module,
  // while still tying it to the module's lifetime (so it gets destroyed along
  // with it).
  // Whereas this information is not serialized as part of the lowered
  // module, we still need to provide a valid instance of the
  // BackendDebugInfo class when the lowered module is deserialized.
  // Since the deserialized modules does not need this information,
  // we create a "dummy" instance with no extra code dependencies (to avoid
  // overhead) when the backend is created in __setstate__.
  c10::intrusive_ptr<torch::CustomClassHolder> backend_debug_info_class;
  const c10::QualifiedName backend_debug_info_class_name(
      {"__torch__",
       "torch",
       "classes",
       kBackendUtilsNamespace,
       kBackendDebugInfoClass});
  auto debug_info_cls =
      getCustomClass(backend_debug_info_class_name.qualifiedName());
  TORCH_CHECK(debug_info_cls, "BackendDebugInfo class must be available.");
  loweredModule.register_attribute(
      "__backend_debug_info",
      OptionalType::create(debug_info_cls),
      IValue::make_capsule(backend_debug_info_class));
  static const auto create_backend_debug_info_ct = CodeTemplate(R"(
            def __create_backend_debug_info(self):
                self.__backend_debug_info = $backend_debug_info()
            )");
  TemplateEnv create_backend_debug_info_te;
  create_backend_debug_info_te.s(
      "backend_debug_info", backend_debug_info_class_name.qualifiedName());
  loweredModule.define(
      create_backend_debug_info_ct.format(create_backend_debug_info_te),
      loweredModuleResolver());

  // getstate and setstate are for serialization/deserialization of
  // the LoweredModule.
  // setstate is in charge of initializing self.__backend by invoking
  // __create_backend().
  loweredModule.define(
      R"(
            def __getstate__(self):
                # The third parameter indicates whether __setstate__ must create
                # the backend instance. It's hardcoded to True since the only
                # case it can be false is when __setstate__ is called from
                # outside the module (at module creation time), because
                # __create_backed has been called already (also directly).
                return self.__method_compile_spec, self.__processed_module, True
            )",
      loweredModuleResolver());

  loweredModule.define(
      R"(
            def __setstate__(self, state):
                self.__method_compile_spec = state[0]
                self.__processed_module = state[1]
                # state[2] indicates whether to create the backend instance.
                if state[2]:
                    self.__create_backend()
                    self.__create_backend_debug_info()
                if self.__backend.is_available() :
                    self.__handles = self.__backend.compile(self.__processed_module, self.__method_compile_spec)
                else:
                    raise Exception("Backend is not available.")
            )",
      loweredModuleResolver());

  // This loop generates one method on the LoweredModule for every key
  // in method_compile_spec.
  for (auto& e : method_compile_spec) {
    std::string method_name = e.key().toStringRef();
    static const auto method_ct = CodeTemplate(R"(
            def $method(self${,def_inputs}):
                typed_inputs: List[Any] = [${fwd_inputs,}]
                if self.__backend.is_available() :
                  $unpack, = self.__backend.execute(self.__handles["$method"], typed_inputs)
                  ${refine,}
                  return $ret
                else:
                  raise Exception("Backend is not available.")
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
        default_value->repr(
            def_ss, [](std::ostream&, const IValue&) -> bool { return false; });
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
      type_check_ss << tuple_elements[0]->annotation_str() << ")";
      type_checks.emplace_back(type_check_ss.str());
      for (unsigned i = 1, e = tuple_elements.size(); i < e; ++i) {
        type_check_ss.str(std::string());
        type_check_ss.clear();
        out_ss << ", _" << i;
        type_check_ss << "assert isinstance(_" << i << ", "
                      << tuple_elements[i]->annotation_str() << ")";
        type_checks.emplace_back(type_check_ss.str());
      }
    } else {
      type_check_ss << out_ty->annotation_str() << ")";
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

    loweredModule.define(method_ct.format(method_te), loweredModuleResolver());
  }

  // If backend is available, call __setstate__ to ensure that the returned
  // Module is ready to run.
  // Otherwise throw a warning indicating that the resulting Module is not
  // ready for execution until is loaded to a device with the backend.
  loweredModule.run_method("__create_backend");
  if (loweredModule.run_method("__is_available").toBool()) {
    auto state = at::ivalue::Tuple::create(
        method_compile_spec,
        loweredModule.attr("__processed_module"),
        /*create_backend*/ false);
    loweredModule.run_method("__setstate__", state);
  } else {
    TORCH_WARN(
        "Backend [",
        backend_name,
        "] is not available. Execution of this Module is still possible by "
        "saving and loading on a device where the backend is available.");
  }

  // stop debug info recording and get debug_info_map
  auto debug_info_map = debug_info_recorder.stopRecording();
  loweredModule.run_method("__create_backend_debug_info");
  auto backend_debug_info = loweredModule.attr("__backend_debug_info")
                                .toCustomClass<PyTorchBackendDebugInfo>();
  backend_debug_info->setDebugInfoMap(std::move(debug_info_map));

  return loweredModule;
}
} // namespace detail
} // namespace jit
} // namespace torch
