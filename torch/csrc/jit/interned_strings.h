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
_(Neg) \
_(Sigmoid) \
_(Tanh) \
_(Constant) \
_(Undefined) \
_(FusionGroup) \
_(Split) \
_(AddConstant) \
_(Transpose) \
_(Concat) \
_(Reshape) \
_(split) \
_(Dim) \
_(Offset) \
_(value) \
_(Subgraph) \
_(SpatialBN) \
_(Conv) \
_(Caffe2ConvTranspose) \
_(ConvTranspose) \
_(is_test) \
_(epsilon) \
_(order) \
_(momentum) \
_(consumed_inputs) \
_(kernels) \
_(kernel_shape) \
_(kernel) \
_(strides) \
_(stride) \
_(pads) \
_(pad) \
_(dilations) \
_(dilation) \
_(broadcast) \
_(axis) \
_(perm) \
_(shape) \
_(group)

enum BuiltinSymbol {
  #define DEFINE_SYMBOL(s) \
    k##s,
  FORALL_BUILTIN_SYMBOLS(DEFINE_SYMBOL)
  #undef DEFINE_SYMBOL
  kLastSymbol, //where we start counting for new symbols
};

const char * symbolToString(Symbol s);
Symbol stringToSymbol(const std::string & s);

}}
