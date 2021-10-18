#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/custom_class_detail.h>

namespace torch {
namespace jit {
OpCode parseOpCode(const char* str);
using c10::IValue;

IValue expect_field(
    c10::ivalue::TupleElements& elements,
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

namespace {
#define COUNT_OPCODE(_, _a) 1 +
constexpr size_t numOpcodes = FORALL_OPCODES(COUNT_OPCODE) 0;
#undef COUNT_OPCODE

// Pickled strings are memoized, so we can cache a mapping from
// pointers to parsed OpCodes to speed up parsing.
class OpCodeCache {
 private:
  // We store as void* to emphasize that we care only about the
  // address and should not be dereferencing these pointers.
  std::array<const void*, numOpcodes> keys_{};
  std::array<OpCode, numOpcodes> values_{};
  size_t usedEntries_ = 0;

 public:
  OpCodeCache() {
    memset(keys_.data(), 0, keys_.size() * sizeof(keys_[0]));
  }

  OpCode parse(const c10::ivalue::ConstantString& s) {
    const auto endIt = keys_.begin() + usedEntries_;
    auto it = std::find_if(
        keys_.begin(), endIt, [&s](const void* k) { return k == &s; });
    if (it == endIt) {
      OpCode result = parseOpCode(s.string().c_str());
      if (usedEntries_ < numOpcodes) {
        keys_[usedEntries_] = &s;
        values_[usedEntries_++] = result;
      }
      return result;
    }
    // NOTE: I tried implementing the transpose heuristic here to
    // speed up the search, but it removed the benefit of this cache.
    return values_[it - keys_.begin()];
  }
};
} // namespace

void parseInstructions(
    const std::string& function_name,
    c10::ivalue::TupleElements&& ins_list,
    c10::ivalue::TupleElements& debug_handles_m_tuple,
    mobile::Function* function) {
  c10::List<int64_t> debug_handles_list;
  if (!debug_handles_m_tuple.empty()) {
    const std::string& debug_info_function_name =
        debug_handles_m_tuple[0].toStringRef();
    TORCH_CHECK(
        debug_info_function_name == function_name,
        "The function names in the bytecode table and the debug info table do not match.");
    IValue& debug_handles_table = debug_handles_m_tuple[1];
    auto debugHandlesTableElements =
        std::move(*std::move(debug_handles_table).toTuple()).elements();
    debug_handles_list = (expect_field(
                              debugHandlesTableElements,
                              "function_debug_handles",
                              BYTECODE_INDEX_MODULE_DEBUG_HANDLES)
                              .toTuple()
                              ->elements())[0]
                             .toIntList();
    TORCH_CHECK(
        debug_handles_list.size() == ins_list.size(),
        "The numbers of instructions and debug handles strings do not match.");
  }

  // NOTE: this won't perform particularly well if the ins_list IValue
  // didn't come from unpickler and thus have its strings
  // interned. Consider adding a flag to bypass the cache if that
  // becomes an important use case.
  OpCodeCache opCodeCache;
  for (const auto j : c10::irange(ins_list.size())) {
    auto ins_tuple = std::move(ins_list[j]).toTuple();
    c10::ArrayRef<IValue> ins_item = ins_tuple->elements();
    TORCH_CHECK(
        ins_item.size() == 3,
        "There should be three parts in an instruction. The function name is ",
        function_name);
    OpCode op_code = opCodeCache.parse(*ins_item[0].toString());
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
    const c10::ivalue::TupleElements& consts_list,
    mobile::Function* function) {
  for (const auto& constant : consts_list) {
    function->append_constant(constant);
  }
}

void parseTypes(
    const c10::ivalue::TupleElements& types_list,
    mobile::Function* function) {
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

void parseRegisterSize(size_t rsize, mobile::Function* function) {
  function->set_register_size(rsize);
}

} // namespace mobile
} // namespace jit
} // namespace torch
