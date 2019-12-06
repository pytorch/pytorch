#include <torch/csrc/jit/instruction.h>
#include <iostream>
#include <cstring>

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

char const * toString(OpCode op) {
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
    FORALL_OPCODES(OP_INFO)
#undef OP_INFO
  }
  return nullptr;
}

static constexpr size_t instruction_size = 8;
static_assert(sizeof(Instruction) == instruction_size, "Instructions should be 8 bytes");
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

static constexpr char *strOpCode[] = {
#define STR_OP(x, _) #x,
    FORALL_OPCODES(STR_OP)
#undef STR_OP
};

OpCode parseOpCode(const char *str) {
  const int n = sizeof(strOpCode) / sizeof(strOpCode[0]);
  for (int i = 0; i < n; ++i)
  {
    if (strcmp(strOpCode[i], str) == 0)
      return (OpCode) i;
  }
  return OP;
}

bool isOpSupportedInMobile(OpCode op) {
  static constexpr OpCode supported_ops_in_mobile[] {
      OP, OPN, LOAD, MOVE, STOREN, STORE, DROP, DROPR, LOADC, JF, JMP, LOOP, RET, GET_ATTR, SET_ATTR
  };

  for (auto sop : supported_ops_in_mobile) {
    if (op == sop) return true;
  }
  return false;
}

}
}
