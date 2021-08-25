#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <cstring>
#include <iostream>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, OpCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return out << #x;
    FORALL_OPCODES(OP_STRING)
#undef OP_STRING
  }
  return out;
}

char const* toString(OpCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return #x;
    FORALL_OPCODES(OP_STRING)
#undef OP_STRING
  }
  return nullptr;
}

const char* OpInfo(OpCode op) {
  switch (op) {
#define OP_INFO(x, info) \
  case x:                \
    return info;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    FORALL_OPCODES(OP_INFO)
#undef OP_INFO
  }
  return nullptr;
}

static constexpr size_t instruction_size = 8;
static_assert(
    sizeof(Instruction) == instruction_size,
    "Instructions should be 8 bytes");
std::ostream& operator<<(std::ostream& out, Instruction inst) {
  // TODO: use op info to print out the op in a more user-friendly way
  int nargs = std::strlen(OpInfo(inst.op));
  out << inst.op;
  if (nargs > 0) {
    out << " " << inst.X;
  }
  if (nargs > 1) {
    out << " " << inst.N;
  }
  return out;
}

OpCode parseOpCode(const char* str) {
#define CHECK_OP(op) if (strcmp(str, #op) == 0) { return op; }
  switch (str[0]) {
    case 'C':
      CHECK_OP(CALL);
      CHECK_OP(CREATE_OBJECT);
      break;
    case 'D':
      CHECK_OP(DROP);
      CHECK_OP(DROPR);
      CHECK_OP(DICT_CONSTRUCT);
      break;
    case 'E':
      CHECK_OP(ENTER);
      CHECK_OP(EXIT);
      break;
    case 'F':
      CHECK_OP(FAIL_GUARD);
      CHECK_OP(FORK);
      break;
    case 'G':
      CHECK_OP(GET_ATTR);
      CHECK_OP(GUARD);
      break;
    case 'I':
      CHECK_OP(INTERFACE_CALL);
      CHECK_OP(ISINSTANCE);
      break;
    case 'J':
      CHECK_OP(JF);
      CHECK_OP(JMP);
      break;
    case 'L':
      CHECK_OP(LOAD);
      CHECK_OP(LOADC);
      CHECK_OP(LOOP);
      CHECK_OP(LIST_CONSTRUCT);
      CHECK_OP(LIST_UNPACK);
      break;
    case 'M':
      CHECK_OP(MOVE);
      break;
    case 'N':
      CHECK_OP(NAMED_TUPLE_CONSTRUCT);
      break;
    case 'O':
      // No need to check for OP explicitly; we return it by default!
      CHECK_OP(OPN);
      break;
    case 'P':
      CHECK_OP(PROFILE_OP);
      break;
    case 'R':
      CHECK_OP(RET);
      break;
    case 'S':
      CHECK_OP(STORE);
      CHECK_OP(STOREN);
      CHECK_OP(SET_ATTR);
      break;
    case 'T':
      CHECK_OP(TAIL_CALL);
      CHECK_OP(TUPLE_CONSTRUCT);
      CHECK_OP(TUPLE_SLICE);
      CHECK_OP(TYPECHECK);
      break;
    case 'W':
      CHECK_OP(WAIT);
      CHECK_OP(WARN);
      break;
  }
  #undef CHECK_OP
  return OP;
}

bool isOpSupportedInMobile(OpCode op) {
  // clang-format off
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static constexpr OpCode supported_ops_in_mobile[] {
      OP, OPN, LOAD, MOVE, STOREN, STORE, DROP, DROPR, LOADC, JF, JMP, LOOP,
      RET, GET_ATTR, SET_ATTR, LIST_CONSTRUCT, TUPLE_CONSTRUCT, WARN,
      INTERFACE_CALL, LIST_UNPACK, TUPLE_SLICE, DICT_CONSTRUCT,
      NAMED_TUPLE_CONSTRUCT, CREATE_OBJECT, ISINSTANCE
  };
  // clang-format on

  for (auto sop : supported_ops_in_mobile) {
    if (op == sop)
      return true;
  }
  return false;
}

} // namespace jit
} // namespace torch
