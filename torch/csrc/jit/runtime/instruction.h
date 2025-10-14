#pragma once

#include <cstdint>
#include <typeinfo>
#include <unordered_set>

namespace torch::jit {
// instruction look like:
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
// C - index into code table

#define FORALL_OPCODES(_)                                                      \
  _(OP, "O") /* invoke operator X */                                           \
  _(OPN, "OI") /* invoke vararg operator X with N arguments */                 \
  _(LOAD, "R") /* push a value from a register X */                            \
  _(MOVE, "R") /* push a value from register X, clearing the register */       \
  _(STOREN, "RI") /* store N values to registers [X, X+N) */                   \
  _(STORE, "R") /* store 1 value to registers X */                             \
  _(DROP, "") /* drop 1 value from the top of the stack */                     \
  _(DROPR, "R") /* clear register X */                                         \
  _(LOADC, "C") /* push the constant X */                                      \
  _(JF, "P") /* pop the top of the stack, if false, branch to P */             \
  _(JMP, "P") /* unconditional branch to X */                                  \
  _(LOOP, "PI") /* perform a loop, X is where to branch if cond is false */    \
  _(RET, "") /* exit execution */                                              \
  _(WAIT, "") /* wait for a future to be complete */                           \
  _(CALL, "F") /* call function X */                                           \
  _(GUARD, "T") /* check a guard against type_table, true if passes */         \
  _(TYPECHECK, "TN") /* check each type of input[i] against type_table[X+N] */ \
  _(FAIL_GUARD, "T") /* fail a guard, patch back to GUARD */                   \
  _(PROFILE_OP, "F") /* get a callback from profile_function_table at X */     \
  _(TAIL_CALL, "F") /* replace current frame with function F */                \
  _(INTERFACE_CALL, "CI") /* call method X on the first argument (of N) */     \
  _(GET_ATTR, "S") /* get attribute from slot X in an Object */                \
  _(SET_ATTR, "S") /* set attribute to slot X in an Object */                  \
  _(LIST_UNPACK, "I") /* unpack list expecting length I */                     \
  _(TUPLE_CONSTRUCT, "I") /* construct a tuple using X inputs */               \
  _(NAMED_TUPLE_CONSTRUCT,                                                     \
    "TI") /* construct a tuple of type X, using N inputs */                    \
  _(LIST_CONSTRUCT, "TI") /* construct a list of type X, using N inputs */     \
  _(DICT_CONSTRUCT, "TI") /* construct a dict of type X, using N inputs */     \
  _(CREATE_OBJECT, "T") /* create an object of type X */                       \
  _(ISINSTANCE, "TI") /* check object is one of  types[X:X+N]  */              \
  _(TUPLE_SLICE, "II") /* slice tup[X:(X+N)] */                                \
  _(TUPLE_INDEX, "") /* get the value from a tuple at that index */            \
  _(RAISE_EXCEPTION, "") /* throws the exception from Python */                \
  _(DICT_INDEX, "") /* gets the value from the dict for given key */           \
  _(UNCHECKED_CAST, "") /* perform an unchecked cast operation */              \
  _(__IS__, "") /* performs `is` operator from Python */                       \
  _(UN_INITIALIZED,                                                            \
    "") /* sets default values to variables that are uninitialized */          \
  _(__ISNOT__, "") /* performs `is not` operator from Python  */               \
  _(FORMAT, "I") /* performs string format function `f strings` or `{}.format` \
                     the number of inputs in stored in X */                    \
  _(DEVICE, "") /* invokes aten::device for a Tensor */                        \
  _(DTYPE, "") /* invokes aten::dtype for a Tensor */                          \
  _(DIM, "") /* invokes aten::dim for a Tensor */                              \
  _(__NOT__, "") /* performs `not` operator from Python  */                    \
  _(TO_LIST, "") /* convert the input to a list */                             \
  _(NUM_TO_TENSOR,                                                             \
    "") /* performs the conversion of a number/scalar to Tensor */             \
  _(IS_CUDA, "") /* invokes aten::is_cuda for a Tensor */                      \
  _(FORK, "CN") /* launch a thread to run code entry x with N inputs  */       \
  _(WARN, "I") /* emit a warning with line information */                      \
  _(ENTER, "EN") /* enter scope of a contextmanager */                         \
  _(EXIT, "EX") /* exit the last entered contextmanager */                     \
  _(AWAITABLE, "CN") /* initialize await for code entry x with N inputs  */

enum OpCode : uint8_t {
#define DEFINE_OP(op, _) op,
  FORALL_OPCODES(DEFINE_OP)
#undef DEFINE_OP
};

struct Instruction {
  OpCode op;
  uint8_t unused;
  uint16_t N;
  int32_t X;
  // TODO: check for overflow
  Instruction(OpCode op, int32_t X, uint16_t N)
      : op(op), unused(0), N(N), X(X) {}
};
std::ostream& operator<<(std::ostream& out, Instruction inst);

bool isOpSupportedInMobile(OpCode op);
char const* toString(OpCode op);
OpCode parseOpCode(const char* str);

} // namespace torch::jit
