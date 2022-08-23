#pragma once

#include <cstddef>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

namespace {

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

TensorView* makeContigConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .shape(shape)
      .dtype(dtype)
      .contiguity(std::vector<bool>(shape.size(), true))
      .build();
}

void checkIntValue(
    ExpressionEvaluator& evaluator,
    Val* val,
    Int::ScalarType expected_value) {
  TORCH_CHECK(val->isAnInt());
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

void checkIntValue(
    kir::ExpressionEvaluator& evaluator,
    const Val* val,
    Int::ScalarType expected_value) {
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

// prime numbers
int64_t prime_numbers[] = {
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223};

} // namespace
} // namespace jit
} // namespace torch
