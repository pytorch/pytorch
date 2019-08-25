#include <torch/csrc/jit/instruction.h>
#include <iostream>
#include <cstring>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, OperatorCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return out << #x;
    FORALL_OPCODES(OP_STRING)
#undef OP_STRING
  }
  return out;
}

const char* OpInfo(OperatorCode op) {
  switch (op) {
#define OP_INFO(x, info) \
  case x:                \
    return info;
    FORALL_OPCODES(OP_INFO)
#undef OP_INFO
  }
  return nullptr;
}

static_assert(sizeof(Instruction) == 8, "Instructions should be 8 bytes");
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
}
}
