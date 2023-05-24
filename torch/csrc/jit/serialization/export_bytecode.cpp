#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <utility>

#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/method.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

namespace torch::jit {

std::vector<Method> gatherGetSetStates(ObjectPtr obj) {
  std::vector<Method> methods;
  // Use DFS on IValue's to traverse dependencies of module._ivalue and
  // add all setstate/getstates to initial stack.
  std::vector<ObjectPtr> ivalue_stack;
  ivalue_stack.emplace_back(obj);
  while (!ivalue_stack.empty()) {
    ObjectPtr cur = ivalue_stack.back();
    ivalue_stack.pop_back();
    auto type = cur->type();
    Function* setstate = type->findMethod("__setstate__");
    Function* getstate = type->findMethod("__getstate__");
    if (getstate && setstate) {
      if (setstate->isGraphFunction()) {
        methods.emplace_back(cur, setstate);
      }
      if (getstate->isGraphFunction()) {
        methods.emplace_back(cur, getstate);
      }
    } else {
      for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
        IValue field = cur->getSlot(i);
        if (field.isObject()) {
          ivalue_stack.emplace_back(field.toObject());
        }
      }
    }
  }
  return methods;
}

std::vector<Method> findAllDependentFunctions(
    const Module& module,
    Graph& graph) {
  std::vector<Method> methods;
  std::unordered_set<c10::string_view> called_method_names;
  auto nodes = findAllNodes(graph, c10::prim::CallMethod, true);
  for (Node* node : nodes) {
    if (auto iface = node->input(0)->type()->castRaw<InterfaceType>()) {
      const FunctionSchema* schema = iface->getMethod(node->s(attr::name));
      called_method_names.insert(schema->name());
    }
  }

  for (const auto& submodule : module.modules()) {
    for (const auto& m : submodule.get_methods()) {
      if (called_method_names.find(m.function().qualname().name()) !=
          called_method_names.end()) {
        methods.emplace_back(m);
      }
    }
  }
  return methods;
}

// NOTE: order of functions returned will be:
// 1. functions originated from the methods passed in will be first
// 2. All the dependent functions will come afterwards.
// This order is meaningful because currently mobile Module looks up
// methods with linear search.
std::vector<std::unique_ptr<GraphFunction>> inlineFunctions(
    const std::vector<Method>& initial_methods,
    bool incl_dependent_functions) {
  std::set<std::pair<std::string, Function*>> visited;
  std::deque<Method> stack;
  std::copy(
      initial_methods.begin(),
      initial_methods.end(),
      std::back_inserter(stack));
  std::vector<std::unique_ptr<GraphFunction>> inlined_functions;
  while (!stack.empty()) {
    Method cur = stack.front();
    stack.pop_front();
    auto tup = std::make_pair(
        cur.owner()._ivalue()->type()->name()->qualifiedName(),
        &cur.function());
    if (visited.find(tup) != visited.end()) {
      continue;
    }
    visited.insert(tup);
    const auto& f = toGraphFunction(cur.function());
    auto graph = f.graph()->copyUnique();
    Inline(*graph);
    c10::QualifiedName qn(*cur.owner()._ivalue()->type()->name(), f.name());

    if (incl_dependent_functions) {
      std::vector<Method> dependent_methods =
          findAllDependentFunctions(cur.owner(), *graph);
      std::copy(
          dependent_methods.begin(),
          dependent_methods.end(),
          std::back_inserter(stack));
    }
    auto inlined_func = std::make_unique<GraphFunction>(
        qn, std::move(graph), f.function_creator());
    inlined_func->setSchema(f.getSchema());
    inlined_functions.emplace_back(std::move(inlined_func));
  }
  return inlined_functions;
}

mobile::Code compileGraphToMobileCode(
    const std::string& name,
    const std::shared_ptr<Graph>& graph,
    const CompilationOptions& compilation_options,
    BackendDebugInfoRecorder& debug_info_recorder) {
  MobileCode code(
      graph,
      name,
      compilation_options.enable_default_value_for_unspecified_arg,
      compilation_options.enable_default_args_before_out_args,
      compilation_options.enable_emit_promoted_ops);

  mobile::Code mobile_code;

  // operator names
  std::vector<std::string> method_names;
  std::vector<int64_t> op_debug_handles;
  int next_new_op_index = 0;

  auto op_to_specified_args = code.op_to_num_specified_args();

  for (size_t i = 0; i < code.instructions().size(); ++i) {
    Instruction ins = code.instructions()[i];

    if ((ins.op == OP || ins.op == OPN) && ins.X == next_new_op_index) {
      // Found a new op (assumes new operators ordered by ascending ins.X)
      auto node = code.instructions_source()[i];
      const c10::OperatorName& opname = node->schema().operator_name();
      auto unique_name = c10::toString(opname);
      // For operator with vararg, adding default arguments would be confusing
      // and is not allowed. For an operator with num_args = -1, it means the
      // number of arguments is not available for this operator, we don't do any
      // backward compatibility adaptation at runtime.
      c10::optional<int> num_args = c10::nullopt;
      auto it = op_to_specified_args.find(unique_name);
      if (it != op_to_specified_args.end()) {
        num_args = it->second;
      }
      mobile_code.operator_input_sizes_.emplace_back(num_args.value_or(-1));
      mobile_code.op_names_.emplace_back(opname);
      auto func = mobile::makeOperatorFunction(opname, num_args);
      TORCH_INTERNAL_ASSERT(
          func.has_value(),
          "Operator with name: ",
          toString(opname),
          " not found");
      mobile_code.operators_.emplace_back(*func);
      next_new_op_index++;
    }
    // CALL nodes at this point represent built-in (i.e. non-Graph)
    // functions that were not inlined. Here we convert the CALL
    // instructions for these functions into INTERFACE_CALL instructions
    // s.t. at runtime, we will look up the Function* on the Type of the
    // 0th argument in the stack and call that directly.
    if (ins.op == CALL) {
      auto node = code.instructions_source()[i];
      if (node->kind() == prim::CallMethod) {
        // NB: replacing instruction
        auto method_name_idx =
            code.constant_table().size() + method_names.size();
        method_names.emplace_back(node->s(attr::name));
        ins = Instruction{
            INTERFACE_CALL,
            static_cast<int32_t>(method_name_idx),
            static_cast<uint16_t>(node->inputs().size())};
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      auto node = code.instructions_source()[i];
      for (const auto& input : node->inputs()) {
        const auto& input_type = input->type();
        if (input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          for (const TypePtr& element_type : input_type->containedTypes()) {
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returning a list or dictionary with pytorch class type ",
                "is not supported in mobile module "
                "(List[Foo] or Dict[int, Foo] for class Foo(torch.nn.Module)). "
                "Workaround: instead of using pytorch class as their element type, ",
                "use a combination of list, dictionary, and single types.");
          }
        }
      }
    } else {
      TORCH_CHECK(
          isOpSupportedInMobile(ins.op),
          toString(ins.op),
          " is not supported in mobile module.");
    }
    auto node = code.instructions_source()[i];
    int64_t debug_handle = debug_info_recorder.getNextDebugHandle(node);
    // Note 1-to-1 correspondence between instructions and debug handles
    mobile_code.instructions_.emplace_back(ins);
    mobile_code.debug_handles_.emplace_back(debug_handle);
  }

  // copy constants
  mobile_code.constants_ = code.constant_table();

  // Make a copy of the constants and append the method names
  // that we emitted for the converted INTERFACE_CALL nodes above.
  for (auto& method_name : method_names) {
    mobile_code.constants_.emplace_back(method_name);
  }

  mobile_code.types_ = code.type_table();
  mobile_code.register_size_ = code.register_size();
  return mobile_code;
}

std::unique_ptr<mobile::Function> convertJitFunctionToMobileFunction(
    const GraphFunction& function,
    const CompilationOptions& options) {
  BackendDebugInfoRecorder debug_handle;
  auto mobileCode = compileGraphToMobileCode(
      function.name(), function.graph(), options, debug_handle);
  const auto& schema = function.getSchema();
  return std::make_unique<mobile::Function>(
      function.qualname(), std::move(mobileCode), schema);
}

IValue convertMobileFunctionToCodeTable(
    const mobile::Function& func,
    const CompilationOptions& compilation_options) {
  auto code = func.get_code();
  std::vector<IValue> instructions;
  instructions.reserve(code.instructions_.size());
  for (Instruction ins : code.instructions_) {
    instructions.emplace_back(to_tuple({toString(ins.op), ins.X, ins.N}));
  }

  std::vector<IValue> operators;
  operators.reserve(code.op_names_.size());
  for (unsigned i = 0; i < code.op_names_.size(); ++i) {
    const auto& opname = code.op_names_[i];
    const int size = code.operator_input_sizes_[i];
    if (compilation_options.enable_default_value_for_unspecified_arg) {
      operators.emplace_back(to_tuple({opname.name, opname.overload_name}));
    } else {
      operators.emplace_back(
          to_tuple({opname.name, opname.overload_name, size}));
    }
  }

  std::vector<IValue> types;
  for (const TypePtr& t : code.types_) {
    std::string type_str = t->annotation_str();
    types.emplace_back(type_str);
  }

  auto register_size = static_cast<int>(code.register_size_);
  auto codeTable = Table(
      {{"instructions", to_tuple(instructions)},
       {"operators", to_tuple(operators)},
       {"constants", to_tuple(code.constants_)},
       {"types", to_tuple(types)},
       {"register_size", register_size}});

  return codeTable;
}

void checkSchema(const c10::FunctionSchema& schema) {
  TORCH_CHECK(
      schema.overload_name().empty(), // @TODO: is this check correct?
      "Overloads are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_vararg(), "Python *args are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_varret(),
      "A variable number of return values is not supported in mobile modules.");
}

bool isLoweredModule(const Module& m) {
  c10::QualifiedName type_name;
  if (m.type()->name()) {
    type_name = m.type()->name().value();
  }
  bool isLoweredModule = false;
  for (const auto& atom : type_name.atoms()) {
    if (atom == "LoweredModule") {
      isLoweredModule = true;
      break;
    }
  }
  return isLoweredModule;
}

// Check if the global static map of backend debug info
// contains debug info for this module and any of its children.
// If so combine all the maps together and return one.
void getBackendDebugInfoMap(
    const Module& m,
    BackendDebugInfoMapType& debug_map) {
  if (isLoweredModule(m)) {
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      debug_map.insert(map.value().begin(), map.value().end());
    }
  }
  for (const auto& c : m.children()) {
    getBackendDebugInfoMap(c, debug_map);
  }
}

uint64_t get_min_operator_version_from_version_map(
    const mobile::Module& module) {
  uint64_t min_version = caffe2::serialize::kMinSupportedFileFormatVersion;
  for (const auto& func : module.compilation_unit().methods()) {
    for (const auto& op_name : func->get_code().op_names_) {
      auto schema_name = op_name.overload_name.empty()
          ? op_name.name
          : op_name.name + "." + op_name.overload_name;
      auto version_entry = get_operator_version_map().find(schema_name);
      if (version_entry != get_operator_version_map().end()) {
        const auto& entry = version_entry->second;
        min_version = std::max(
            min_version, uint64_t(entry[entry.size() - 1].bumped_at_version));
      }
    }
  }
  return min_version;
}

mobile::Module jitModuleToMobile(
    const Module& module,
    const CompilationOptions& options) {
  std::shared_ptr<mobile::CompilationUnit> mcu =
      std::make_shared<mobile::CompilationUnit>();
  BackendDebugInfoRecorder debug_info_recorder;

  std::vector<Method> methods_to_export = module.get_methods();
  std::vector<Method> getsetstates = gatherGetSetStates(module._ivalue());
  std::copy(
      getsetstates.begin(),
      getsetstates.end(),
      std::back_inserter(methods_to_export));

  for (const auto& func :
       inlineFunctions(methods_to_export, options.incl_interface_call)) {
    auto mobile_code = compileGraphToMobileCode(
        func->name(), func->graph(), options, debug_info_recorder);
    const auto& schema = func->getSchema();
    checkSchema(schema);
    auto mobile_func = std::make_unique<mobile::Function>(
        func->qualname(), std::move(mobile_code), schema);
    mcu->register_function(std::move(mobile_func));
  }

  mobile::Module m(module._ivalue(), mcu);
  m.setHasDebugHandles(true);
  BackendDebugInfoMapType backend_debug_info_map;
  getBackendDebugInfoMap(module, backend_debug_info_map);
  auto debug_handle_cs_ptr_map = debug_info_recorder.stopRecording();
  debug_handle_cs_ptr_map.insert(
      backend_debug_info_map.begin(), backend_debug_info_map.end());
  m.setDebugTable(MobileDebugTable(
      debug_handle_cs_ptr_map.begin(), debug_handle_cs_ptr_map.end()));
  m.set_min_operator_version(get_min_operator_version_from_version_map(m));
  m.set_bytecode_version(options.model_version);
  return m;
}

} // namespace torch::jit
