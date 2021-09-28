#include <torch/csrc/jit/serialization/export_bytecode.h>

#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch {
namespace jit {

IValue to_tuple(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(to_tuple({e.first, e.second}));
  }
  return to_tuple(std::move(ivalue_entries));
}

namespace {
std::pair<IValue, IValue> getFunctionTuple(
    const FunctionSchema& schema,
    const MobileCode& code,
    BackendDebugInfoRecorder& debug_info_recorder,
    const std::string& qn,
    TypeNameUniquer& type_name_uniquer_) {
  auto instructions_copy = code.instructions();

  // operator names
  std::vector<c10::OperatorName> opnames;
  std::vector<std::string> method_names;
  std::vector<int64_t> op_debug_handles;
  int next_new_op_index = 0;
  for (size_t i = 0; i < instructions_copy.size(); ++i) {
    Instruction ins = instructions_copy[i];
    if ((ins.op == OP || ins.op == OPN) && ins.X == next_new_op_index) {
      // Found a new op (assumes new operators ordered by ascending ins.X)
      auto node = code.instructions_source()[i];
      opnames.emplace_back(node->schema().operator_name());
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
        Instruction new_instr{
            INTERFACE_CALL,
            static_cast<int32_t>(method_name_idx),
            static_cast<uint16_t>(node->inputs().size())};
        instructions_copy[i] = new_instr;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      auto node = code.instructions_source()[i];
      for (const auto& input : node->inputs()) {
        const auto& input_type = input->type();
        if (input_type->kind() == TypeKind::TupleType) {
          if (const auto& name_typed_input =
                  input_type->cast<at::NamedType>()) {
            TORCH_CHECK(
                !name_typed_input->name(),
                "A named tuple type is not supported in mobile module. ",
                "Workaround: instead of using a named tuple type's fields, ",
                "use a dictionary type's key-value pair itmes or ",
                "a pytorch class (class Foo(torch.nn.Module))'s attributes.'");
          }
        } else if (
            input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          for (const TypePtr& element_type : input_type->containedTypes()) {
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returining a list or dictionary with pytorch class type ",
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
    op_debug_handles.emplace_back(debug_handle);
  }

  // instructions
  std::vector<IValue> instructions;
  instructions.reserve(instructions_copy.size());
  for (Instruction ins : instructions_copy) {
    instructions.emplace_back(to_tuple({toString(ins.op), ins.X, ins.N}));
  }

  // operators
  std::vector<IValue> operators;
  auto op_to_specified_args = code.op_to_num_specified_args();
  operators.reserve(opnames.size());
  for (const auto& opname : opnames) {
    auto unique_name = c10::toString(opname);
    // For operator with vararg, adding default arguments would be confusing and
    // is not allowed. For an operator with num_args = -1, it means the number
    // of arguments is not available for this operator, we don't do any backward
    // compatibility adaptation at runtime.
    int num_args = -1;
    auto it = op_to_specified_args.find(unique_name);
    if (it != op_to_specified_args.end()) {
      num_args = it->second;
    }
    if (BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled()) {
      operators.emplace_back(to_tuple({opname.name, opname.overload_name}));
    } else {
      operators.emplace_back(
          to_tuple({opname.name, opname.overload_name, num_args}));
    }
  }

  // constants
  //
  // Make a copy of the constants and append the method names
  // that we emitted for the converted INTERFACE_CALL nodes above.
  auto constants = code.constant_table();
  for (auto& method_name : method_names) {
    constants.emplace_back(std::move(method_name));
  }

  // types
  std::vector<IValue> types;
  types.reserve(code.type_table().size());
  static const std::string torch_prefix("__torch__");
  static const std::string class_prefix("__torch__.torch.classes");
  for (const TypePtr& t : code.type_table()) {
    auto type_str = t->annotation_str();
    if (type_str.find(torch_prefix) == 0) {
      TORCH_CHECK(
          type_str.find(class_prefix) == 0,
          "__torch__ types other than torchbind (__torch__.torch.classes)"
          "are not supported in lite interpreter. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)).");
    }
    types.emplace_back(type_str);
  }

  // since the register location is embedded into the bytecode, pass the
  // register size
  auto register_size = static_cast<int>(code.register_size());

  auto codeTable = Table(
      {{"instructions", to_tuple(instructions)},
       {"operators", to_tuple(operators)},
       {"constants", to_tuple(constants)},
       {"types", to_tuple(types)},
       {"register_size", register_size}});

  // schema
  auto type_printer =
      [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
    auto namedType = t->cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };
  TORCH_CHECK(
      schema.overload_name().empty(), // @TODO: is this check correct?
      "Overloads are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_vararg(), "Python *args are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_varret(),
      "A variable number of return values is not supported in mobile modules.");
  auto makeArgTuple = [&](const std::vector<Argument>& args) {
    std::vector<IValue> argTables;
    for (auto&& arg : args) {
      TORCH_CHECK(
          !arg.N(),
          "Arguments with known list lengths are not supported in mobile modules.");
      TORCH_CHECK(
          !arg.kwarg_only(),
          "Keyword-only arguments are not supported in mobile modules.");
      /*
        This part adds the argument's name, type and default_value in
        `bytecode.pkl` This has to be consistent with the `code/` directory
        which has annotated py code of the entire module. `type_printer` uses
        `TypeNameUniquer` to get the managled name of the argument. This helps
        in having the right object reference when a class method is called using
        the `self` argument.

        arg.type()->annotation_str(type_printer) => mangled unique name of the
        module/submodule
      */
      argTables.emplace_back(Table({
          {"name", arg.name()},
          {"type", arg.type()->annotation_str(type_printer)},
          {"default_value", arg.default_value()},
      }));
    }
    return to_tuple(argTables);
  };
  auto schemaTable = Table({
      {"arguments", makeArgTuple(schema.arguments())},
      {"returns", makeArgTuple(schema.returns())},
  });

  // function tuple
  auto bytecode_vals = to_tuple({qn, codeTable, schemaTable});

  c10::optional<IValue> debug_info_vals;
  // module debug info
  // This is just a set of debug handles.
  // We always save debug handles.
  // debug handles generated by debug_handle_manager
  // will correspond to {source_range, inlinedCallStackPtr} which we will
  // serialize separately.
  IValue module_debug_tuple = c10::ivalue::Tuple::create(op_debug_handles);
  auto function_debug_info =
      Table({{"function_debug_handles", module_debug_tuple}});
  debug_info_vals = to_tuple({qn, function_debug_info});
  return std::make_pair(bytecode_vals, debug_info_vals);
}

} // namespace

void BytecodeExportSet::add(
    const c10::QualifiedName& qn,
    ExportedBytecode exported) {
  items_.emplace(qn, std::move(exported));
}

void BytecodeExportSet::update(const c10::QualifiedName& qn, bool toplevel) {
  items_.at(qn).toplevel = toplevel;
}

bool BytecodeExportSet::contains(const c10::QualifiedName& qn) const {
  return items_.find(qn) != items_.end();
}

void BytecodeExportSet::exportIValues(
    std::vector<c10::IValue>& elements,
    std::vector<c10::IValue>& debugInfoElements,
    BackendDebugInfoRecorder& recorder,
    TypeNameUniquer& uniquer) const {
  size_t toplevelEnd = elements.size() - 1;
  for (const auto& item : items_) {
    auto tuple = getFunctionTuple(
        item.second.schema,
        *item.second.code,
        recorder,
        item.first.qualifiedName(),
        uniquer);
    elements.push_back(std::move(tuple.first));
    debugInfoElements.push_back(std::move(tuple.second));

    // Make sure toplevel methods are exported first for forward compatibility.
    if (item.second.toplevel) {
      toplevelEnd++;
      std::swap(elements[toplevelEnd], elements.back());
      std::swap(debugInfoElements[toplevelEnd], debugInfoElements.back());
    }
  }
}

void BytecodeExportSet::exportIValues(
    std::vector<c10::IValue>& elements,
    BackendDebugInfoRecorder& recorder,
    TypeNameUniquer& uniquer) const {
  std::vector<c10::IValue> debugInfoElements;
  exportIValues(elements, debugInfoElements, recorder, uniquer);
}
} // namespace jit
} // namespace torch
