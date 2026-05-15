#pragma once

#include <ATen/code_template.h>

namespace torch::jit::fuser::cpu {

/*with type_as not checking type of its input, a fusion group can have non-fp32
tensor as input. Correct code for this case is generated, however, nvrtc does
not know how to handle int*_t integer types, so typedefs help it handle those
cases*/

static auto type_declarations_template = at::jit::CodeTemplate(R"(

#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY

typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T* data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");

static auto cpu_compilation_unit_template = at::jit::CodeTemplate(R"(
#include <math.h>
#include <cstddef>
#include <cstdint>

double rsqrt(double x) {
  return 1.0/sqrt(x);
}

float rsqrtf(float x) {
  return 1.0f/sqrtf(x);
}

double frac(double x) {
  return x - trunc(x);
}

float fracf(float x) {
  return x - truncf(x);
}

${type_declarations}

#ifdef _MSC_VER
template<size_t n> struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

#define IndexTypeLoop int_same_size_t<IndexType>
#define ToIndexTypeLoop(x) static_cast<IndexTypeLoop>(x)
#else
#define IndexTypeLoop IndexType
#define ToIndexTypeLoop(x) x
#endif

#define OMP_THRESHOLD 100000
static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
  #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
  for (IndexTypeLoop linearIndex = 0;
        linearIndex < ToIndexTypeLoop(totalElements);
        linearIndex += 1) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}

#ifdef _WIN32
#define JIT_API __declspec(dllexport)
#else
#define JIT_API
#endif

extern "C"
JIT_API void ${kernelName}(IndexType totalElements, void ** args) {
  ${kernelName}_kernel(totalElements ${,argument_loads});
}
)");

} // namespace torch::jit::fuser::cpu
