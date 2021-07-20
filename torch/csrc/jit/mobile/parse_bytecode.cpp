#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/custom_class_detail.h>

namespace torch {
namespace jit {
OpCode parseOpCode(const char* str);
using c10::IValue;

IValue expect_field(
    IValue tup,
    const std::string& expected_name,
    size_t entry) {
  auto row = tup.toTuple()->elements().at(entry).toTuple();
  TORCH_INTERNAL_ASSERT(
      row->elements().at(0).toStringRef() == expected_name,
      "Expected ",
      expected_name,
      " found ",
      row->elements().at(0).toStringRef());
  return row->elements().at(1);
}

IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

namespace mobile {

namespace {} // namespace

void parseInstructions(
    const std::string& function_name,
    const IValue& codeTable,
    const IValue& debug_handles_element,
    mobile::Function* function) {
  const auto& ins_list =
      expect_field(codeTable, "instructions", BYTECODE_INDEX_INSTRUCTION)
          .toTuple()
          ->elements();

  std::vector<IValue> debug_handles_list;
  bool has_debug_handle = !debug_handles_element.isNone();
  if (has_debug_handle) {
    const auto& debug_handles_m_tuple =
        debug_handles_element.toTuple()->elements();
    const std::string& debug_info_function_name =
        debug_handles_m_tuple[0].toStringRef();
    TORCH_CHECK(
        debug_info_function_name == function_name,
        "The function names in the bytecode table and the debug info table do not match.");
    IValue debug_handles_table = debug_handles_m_tuple[1];
    debug_handles_list = (expect_field(
                              debug_handles_table,
                              "function_debug_handles",
                              BYTECODE_INDEX_MODULE_DEBUG_HANDLES)
                              .toTuple()
                              ->elements())[0]
                             .toList()
                             .vec();
    TORCH_CHECK(
        debug_handles_list.size() == ins_list.size(),
        "The numbers of instructions and debug handles strings do not match.");
  }

  for (size_t i = 0; i < ins_list.size(); ++i) {
    auto ins_item = ins_list[i].toTuple()->elements();
    TORCH_CHECK(
        ins_item.size() == 3,
        "There should be three parts in an instruction. The function name is ",
        function_name);
    OpCode op_code = parseOpCode(ins_item[0].toString()->string().c_str());
    int X = ins_item[1].toInt();
    int N = ins_item[2].toInt();
    if (has_debug_handle) {
      int64_t debug_handle = debug_handles_list[i].toInt();
      function->append_instruction(op_code, X, N, debug_handle);
    } else {
      function->append_instruction(op_code, X, N);
    }
  }
}

void parseConstants(const IValue& codeTable, mobile::Function* function) {
  const auto& consts_list =
      expect_field(codeTable, "constants", BYTECODE_INDEX_CONSTANT)
          .toTuple()
          ->elements();
  for (const auto& constant : consts_list) {
    function->append_constant(constant);
  }
}

void parseTypes(const IValue& codeTable, mobile::Function* function) {
  const auto& types_list = expect_field(codeTable, "types", BYTECODE_INDEX_TYPE)
                               .toTuple()
                               ->elements();
  static const c10::QualifiedName classPrefix = "__torch__.torch.classes";
  for (const auto& t : types_list) {
    c10::QualifiedName qn(t.toStringRef());
    if (classPrefix.isPrefixOf(qn)) {
      auto classType = getCustomClass(qn.qualifiedName());
      TORCH_CHECK(
          classType,
          "The implementation of class ",
          qn.qualifiedName(),
          " cannot be found.");
      function->append_type(classType);
    } else {
      function->append_type(c10::parseType(t.toStringRef()));
    }
  }
}

void parseRegisterSize(const IValue& codeTable, mobile::Function* function) {
  const auto& register_size =
      expect_field(codeTable, "register_size", BYTECODE_INDEX_REGISTER_SIZE)
          .toInt();
  function->set_register_size(register_size);
}

} // namespace mobile
} // namespace jit
} // namespace torch
