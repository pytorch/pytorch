#pragma once
#include <typeinfo>
#include <stdint.h>
#include <unordered_set>

namespace torch {
namespace jit {
// instructs look like:
// op_code X, N
// meaning of X, N depend on the op:
// O - index into operator table
// R - index into register table
// I - literal integer
// C - index into constant table
// P - jump offset relative to beginning of current instruction
// F - index into function table
// T - index into the type table, used for guard instructions
// S - index into object slots

#define FORALL_OPCODES(_)                                                   \
  _(OP, "O") /* invoke operator X */                                        \
  _(LOAD, "R") /* push a value from a register X */                         \
  _(MOVE, "R") /* push a value from register X, clearing the register */    \
  _(STOREN, "RI") /* store N values to registers [X, X+N) */                \
  _(STORE, "R") /* store 1 value to registers X */                          \
  _(DROP, "") /* drop 1 value from the top of the stack */                  \
  _(DROPR, "R") /* clear register X */                                      \
  _(LOADC, "C") /* push the constant X */                                   \
  _(JF, "P") /* pop the top of the stack, if false, branch to P */          \
  _(JMP, "P") /* unconditional branch to X */                               \
  _(LOOP, "PI") /* perform a loop, X is where to branch if cond is false */ \
  _(RET, "") /* exit execution */                                           \
  _(WAIT, "") /* wait for a future to be complete */                        \
  _(CALL, "F") /* call function X */                                        \
  _(GUARD, "T") /* check guard against type_table, true if passes */        \
  _(TAIL_CALL, "F") /* replace current frame with function F */             \
  _(INTERFACE_CALL, "CI") /* call method X on the first argument (of N) */  \
  _(GET_ATTR, "S") /* get attribute from slot X in an Object */             \
  _(SET_ATTR, "S") /* set attribute to slot X in an Object */

enum OpCode : uint8_t {
#define DEFINE_OP(op, _) op,
  FORALL_OPCODES(DEFINE_OP)
#undef DEFINE_OP
};

struct Instruction {
  OpCode op;
  uint8_t padding; // currently unused
  uint16_t N;
  int32_t X;
  // TODO: check for overflow
  Instruction(OpCode op, int32_t X, uint16_t N)
      : op(op), padding(0), N(N), X(X) {}
};

bool isOpSupportedInMobile(OpCode op);

}
}
