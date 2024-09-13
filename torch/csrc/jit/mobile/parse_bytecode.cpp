#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/custom_class_detail.h>

namespace torch::jit {
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
  return std::move(row)->elements().at(1);
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

void applyUpgrader(mobile::Function* function, uint64_t operator_version) {
  Code& code = function->get_code();
  auto& operator_version_map = getOperatorVersionMapForMobile();
  for (size_t i = 0; i < code.instructions_.size(); i++) {
    Instruction& inst = code.instructions_[i];
    if (inst.op == OpCode::OP) {
      std::string operator_name = code.op_names_[inst.X].name +
          (code.op_names_[inst.X].overload_name.empty()
               ? ""
               : "." + code.op_names_[inst.X].overload_name);

      auto it = operator_version_map.find(operator_name);
      // Find out if there is an upgrader for this operator
      if (it != operator_version_map.end()) {
        auto upgrader_list = it->second;
        // Loop all upgraders for this operator, and find out if there exists a
        // valid upgrader. Use iteration here instead of other faster search
        // algorithm, because the number of upgrader per operator will be just a
        // few and tend to keep the code light-weight from binary size concern.
        for (const auto& upgrader : upgrader_list) {
          if (static_cast<int>(operator_version) <= upgrader.max_version &&
              static_cast<int>(operator_version) >= upgrader.min_version) {
            // If there exists a valid upgrader, change the instruction OP to
            // CALL, and the index will point to the according upgrader
            // function. All upgrader function are available in
            // function->get_code().functions_. It's a vector of function
            // pointer and they are initialized in the same order as the global
            // vector kUpgraderBytecode.
            // Instruction new_inst = inst;
            // new_inst.op = OpCode::CALL;
            // new_inst.X = upgrader.index;
            // code->instructions_[i] = new_inst;
            TORCH_CHECK(
                upgrader.index < static_cast<int>(code.functions_.size()),
                "upgrader index is, ",
                upgrader.index,
                " and it's larger than the upgrader function list length ",
                code.functions_.size());
            inst.op = OpCode::CALL;
            inst.X = upgrader.index;
          }
        }
      }
    }
  }
}

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
                              .toTupleRef()
                              .elements())[0]
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
    auto X = ins_item[1].toInt();
    auto N = ins_item[2].toInt();

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
  std::vector<std::string> types_string_list;
  types_string_list.resize(types_list.size());
  for (size_t i = 0; i < types_list.size(); i++) {
    types_string_list[i] = types_list[i].toStringRef();
  }

  std::vector<c10::TypePtr> types_ptr_list = c10::parseType(types_string_list);
  for (auto& type_ptr : types_ptr_list) {
    function->append_type(type_ptr);
  }
}

void parseRegisterSize(size_t rsize, mobile::Function* function) {
  function->set_register_size(rsize);
}

} // namespace mobile
} // namespace torch::jit
