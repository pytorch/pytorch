/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can signal to template
 * code the type of epilogue we want to run, and let the underlying code specify the details such as
 * element types, accumulator type and elements per vector access.
 *
 */

#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass_extensions/epilogue/thread/ft_fused_activations.h"

namespace fastertransformer {

struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpNoBias {};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasFtGelu> {
    using Op = cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor,
                                                                   ElementType,
                                                                   ElementsPerVectorAccess,
                                                                   ElementAccumulator,
                                                                   ElementAccumulator,
                                                                   cutlass::epilogue::thread::ScaleType::NoBetaScaling,
                                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                                   true>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpNoBias> {
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::Default>;
};

}  // namespace fastertransformer