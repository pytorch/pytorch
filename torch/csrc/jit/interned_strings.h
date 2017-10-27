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
_(Div) \
_(Mul) \
_(Neg) \
_(Sub) \
_(Pow) \
_(Sigmoid) \
_(Tanh) \
_(Constant) \
_(Concat) \
_(Slice) \
_(Squeeze) \
_(Undefined) \
_(FusionGroup) \
_(Split) \
_(Gemm) \
_(AddConstant) \
_(SubConstant) \
_(Scale) \
_(Transpose) \
_(Reshape) \
_(split) \
_(Offset) \
_(value) \
_(Subgraph) \
_(SpatialBN) \
_(Conv) \
_(Caffe2ConvTranspose) \
_(ConvTranspose) \
_(is_test) \
_(epsilon) \
_(expand) \
_(order) \
_(momentum) \
_(consumed_inputs) \
_(kernels) \
_(kernel_shape) \
_(kernel) \
_(scale) \
_(strides) \
_(stride) \
_(pads) \
_(pad) \
_(beta) \
_(alpha) \
_(dilations) \
_(dilation) \
_(broadcast) \
_(axis) \
_(size) \
_(perm) \
_(shape) \
_(axes) \
_(group) \
_(inplace)

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
