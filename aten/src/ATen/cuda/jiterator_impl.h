#pragma once
#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <c10/util/variant.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/JitLoops.cuh>

#include <string>
#include <vector>

namespace at {
namespace native {

constexpr int NUM_INPUTS = 8;

#define AT_FOR_8_INPUTS(_)  \
  _(1)                      \
  _(2)                      \
  _(3)                      \
  _(4)                      \
  _(5)                      \
  _(6)                      \
  _(7)                      \
  _(8)

c10::SmallVector<std::string> get_extra_args_typenames(const std::vector<at::Scalar>& extra_args) {
  c10::SmallVector<std::string> args_typenames(extra_args.size());
  for (auto i = 0; i < extra_args.size(); ++i) {
    args_typenames[i] = at::cuda::jit::typeName(extra_args[i].type());
  }
  return args_typenames;
}

int can_vectorize_up_to(at::ScalarType type, char* pointer) {
  switch(type) {
#define DEFINE_CASE(ctype, scalartype)                                   \
    case ScalarType::scalartype : return memory::can_vectorize_up_to<ctype>(pointer);

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
#undef DEFINE_CASE

    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

// jitted version of the above
// See Note [Jiterator], this relies on the assumptions enumerated there
int jitted_can_vectorize_up_to(const TensorIteratorBase& iter) {
  const at::ScalarType common_dtype = iter.common_dtype();
  const at::ScalarType result_dtype = common_dtype;

  // Deals with output
  int result = can_vectorize_up_to(result_dtype, static_cast<char*>(iter.data_ptr(0)));

  // Incorporates input(s)
  for (auto i = 1; i < iter.ntensors(); ++i) {
    result = std::min<int>(result, can_vectorize_up_to(common_dtype, static_cast<char*>(iter.data_ptr(i))));
  }

  return result;
}

template<int N>
static std::unique_ptr<OffsetCalculator<N>> make_unique_input_offset_calculator(const TensorIteratorBase& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return std::make_unique<OffsetCalculator<N>>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

struct OffsetCalculatorVariant {
#define DEFINE_CASE(index) std::unique_ptr<OffsetCalculator<index>>,
  using OffsetCalculatorTypes = c10::variant<
    AT_FOR_8_INPUTS(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  OffsetCalculatorVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    switch(arity) {
#define DEFINE_CASE(index)        \
      case index : v = make_unique_input_offset_calculator<index>(iter); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE
      default:
        TORCH_CHECK(false, "OffsetCalculatorVariant is not implemented for ninputs = ", arity);
    }
  }

  void* data_ptr() {
    return c10::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

 private:
  OffsetCalculatorTypes v;
};

struct ArrayVariant {
  // notice: This would produce c10::variant<at::detail::Array<char*, 2...9>>
#define DEFINE_CASE(index) at::detail::Array<char*, index + 1>,
  using ArrayTypes = c10::variant<
    AT_FOR_8_INPUTS(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  ArrayVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    // This assumes that jiterator kernels only have 1 output
    switch(arity) {
#define DEFINE_CASE(index)                              \
      case index: array = at::detail::Array<char*, index + 1>{}; break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "ArrayVariant is not implemented for ninputs = ", arity);
    }

    c10::visit([&](auto& a) {
      for (auto i = 0; i < arity + 1; ++i) {
        a[i] = (char*)iter.data_ptr(i);
      }
    }, array);
  }

  void* data_ptr() {
    return c10::visit([](auto & a){ return static_cast<void*>(&a); }, array);
  }

private:
  ArrayTypes array;
};

struct TrivialOffsetCalculatorVariant {
#define DEFINE_CASE(index) TrivialOffsetCalculator<index>,
  using TrivialOffsetCalculatorTypes = c10::variant<
    AT_FOR_8_INPUTS(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  TrivialOffsetCalculatorVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    switch(arity) {
#define DEFINE_CASE(index)      \
      case index: v = TrivialOffsetCalculator<index>(); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "TrivialOffsetCalculatorVariant is not implemented for ninputs = ", arity);
    }
  }

  void* data_ptr() {
    return c10::visit([](auto & v){ return static_cast<void*>(&v); }, v);
  }

private:
  TrivialOffsetCalculatorTypes v;
};

struct LoadWithCastVariant {
#define DEFINE_CASE(index) std::unique_ptr<memory::LoadWithCast<index>>,
  using LoadWithCastPtr = c10::variant<
    AT_FOR_8_INPUTS(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  LoadWithCastVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    switch(arity) {
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::LoadWithCast<index>>(iter); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "LoadWithCastVariant is not implemented for ninputs = ", arity);
    }
  }

  void* data_ptr() {
    return c10::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

private:
  LoadWithCastPtr v;
};

}} // namespace at::native


#endif // AT_USE_JITERATOR()
