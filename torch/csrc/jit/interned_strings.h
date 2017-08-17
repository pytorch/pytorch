#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>

namespace torch { namespace jit {

using Symbol = uint32_t;

#define FORALL_BUILTIN_SYMBOLS(_) \
_(PythonOp) \
_(CppOp) \
_(Param) \
_(Select) \
_(Return) \
_(Eval) \
_(Add) \
_(Mul) \
_(Negate) \
_(Sigmoid) \
_(Tanh) \
_(Constant) \
_(FusionGroup) \
_(Chunk) \
_(NumChunks) \
_(Dim)

enum BuiltinSymbol {
  #define DEFINE_SYMBOL(s) \
    k##s,
  FORALL_BUILTIN_SYMBOLS(DEFINE_SYMBOL)
  #undef DEFINE_SYMBOL
  kLastSymbol, //where we start counting for new symbols
};

const std::string & symbolToString(Symbol s);
Symbol stringToSymbol(const std::string & s);

}}
