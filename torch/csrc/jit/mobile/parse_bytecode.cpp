#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/custom_class_detail.h>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
TypePtr parseCustomType(IValue custom_type);
} // namespace c10

namespace torch {
namespace jit {
OpCode parseOpCode(const char* str);
using c10::IValue;

IValue expect_field(
    std::vector<IValue>& elements,
    const std::string& expected_name,
    size_t entry) {
  auto row = std::move(elements.at(entry)).toTuple();
  TORCH_INTERNAL_ASSERT(
      row->elements().at(0).toStringRef() == expected_name,
      "Expected ",
      expected_name,
      " found ",
      row->elements().at(0).toStringRef());
  return std::move(row->elements().at(1));
}

namespace mobile {

namespace {} // namespace

void parseInstructions(
    const std::string& function_name,
    const std::vector<IValue>& ins_list,
    std::vector<IValue>& debug_handles_m_tuple,
    mobile::Function* function) {
  c10::List<int64_t> debug_handles_list;
  if (!debug_handles_m_tuple.empty()) {
    const std::string& debug_info_function_name =
        debug_handles_m_tuple[0].toStringRef();
    TORCH_CHECK(
        debug_info_function_name == function_name,
        "The function names in the bytecode table and the debug info table do not match.");
    IValue& debug_handles_table = debug_handles_m_tuple[1];
    debug_handles_list =
        (expect_field(
             std::move(debug_handles_table).toTuple()->elements(),
             "function_debug_handles",
             BYTECODE_INDEX_MODULE_DEBUG_HANDLES)
             .toTuple()
             ->elements())[0]
            .toIntList();
    TORCH_CHECK(
        debug_handles_list.size() == ins_list.size(),
        "The numbers of instructions and debug handles strings do not match.");
  }

  for (const auto j : c10::irange(ins_list.size())) {
    std::vector<IValue> ins_item =
        std::move(*std::move(ins_list[j]).toTuple()).elements();
    TORCH_CHECK(
        ins_item.size() == 3,
        "There should be three parts in an instruction. The function name is ",
        function_name);
    OpCode op_code = parseOpCode(ins_item[0].toString()->string().c_str());
    int X = ins_item[1].toInt();
    int N = ins_item[2].toInt();
    if (!debug_handles_list.empty()) {
      int64_t debug_handle = debug_handles_list[j];
      function->append_instruction(op_code, X, N, debug_handle);
    } else {
      function->append_instruction(op_code, X, N);
    }
  }
}

void parseConstants(
    const std::vector<IValue>& consts_list,
    mobile::Function* function) {
  for (const auto& constant : consts_list) {
    function->append_constant(constant);
  }
}

void parseTypes(
    const std::vector<IValue>& types_list,
    mobile::Function* function) {
  static const c10::QualifiedName classPrefix = "__torch__.torch.classes";
  for (const auto& t : types_list) {
    if (t.isString()) {
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
    } else {
      TORCH_CHECK(
          t.isTuple(),
          "Custom type should be tuple, but recieve ",
          t.tagKind());
      auto tt = c10::parseCustomType(t);
      function->append_type(tt);
    }
  }
}

void parseRegisterSize(size_t rsize, mobile::Function* function) {
  function->set_register_size(rsize);
}

} // namespace mobile
} // namespace jit
} // namespace torch
