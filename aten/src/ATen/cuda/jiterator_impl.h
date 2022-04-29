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

#define AT_FOR_8_INPUTS(_)  \
  _(0)                      \
  _(1)                      \
  _(2)                      \
  _(3)                      \
  _(4)                      \
  _(5)                      \
  _(6)                      \
  _(7)


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

template<int ...Is>
auto OffsetCalculatorType_List_Impl(std::integer_sequence<int, Is...>) -> c10::variant<std::unique_ptr<OffsetCalculator<Is>>...>;

template<int N>
using OffsetCalculatorType_List = decltype(OffsetCalculatorType_List_Impl(std::make_integer_sequence<int, N>{}));


struct OffsetCalculatorVariant {
  using OffsetCalculatorTypes = OffsetCalculatorType_List<8>;

  OffsetCalculatorVariant(const TensorIteratorBase& iter) {
    int N = iter.ninputs();
    switch(N) {
#define DEFINE_CASE(index)        \
      case index : v = make_unique_input_offset_calculator<index>(iter); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE
      default:
        TORCH_CHECK(false, "OffsetCalculatorVariant not implemented for ninputs = ", N);
    }
  }

  void* data_ptr() {
    return c10::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

 private:
  OffsetCalculatorTypes v;
};

template<int ...Is>
auto ArrayType_List_Impl(std::integer_sequence<int, Is...>) -> c10::variant<at::detail::Array<char*, Is + 2>...>;

template<int N>
using ArrayType_List = decltype(ArrayType_List_Impl(std::make_integer_sequence<int, N>{}));

struct ArrayVariant {
  // notice: This would produce c10::variant<at::detail::Array<char*, 2...10>>
  using ArrayTypes = ArrayType_List<8>;

  ArrayVariant(const TensorIteratorBase& iter) {
    int N = iter.ntensors();
    // jitted kernels must have at least 1 input and 1 output
    switch(N) {
#define DEFINE_CASE(index)      \
      case index + 2: array = at::detail::Array<char*, index + 2>{}; break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "ArrayVariant not implemented for ninputs = ", N);
    }

    c10::visit([&](auto& a) {
      for (auto i = 0; i < N; ++i) {
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

template<int ...Is>
auto TrivialOffsetCalculator_List_Impl(std::integer_sequence<int, Is...>) -> c10::variant<TrivialOffsetCalculator<Is>...>;

template<int N>
using TrivialOffsetCalculator_List = decltype(TrivialOffsetCalculator_List_Impl(std::make_integer_sequence<int, N>{}));

struct TrivialOffsetCalculatorVariant {
  using TrivialOffsetCalculatorTypes = TrivialOffsetCalculator_List<8>;

  TrivialOffsetCalculatorVariant(int arity) {
    switch(arity) {
#define DEFINE_CASE(index)      \
      case index: v = TrivialOffsetCalculator<index>(); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "TrivialOffsetCalculatorVariant not implemented for ninputs = ", arity);
    }
  }

  void* data_ptr() {
    return c10::visit([](auto & v){ return static_cast<void*>(&v); }, v);
  }

private:
  TrivialOffsetCalculatorTypes v;
};

template<int ...Is>
auto LoadWithCastPtr_List_Impl(std::integer_sequence<int, Is...>) -> c10::variant<std::unique_ptr<memory::LoadWithCast<Is>>...>;

template<int N>
using LoadWithCastPtr_List = decltype(LoadWithCastPtr_List_Impl(std::make_integer_sequence<int, N>{}));

struct LoadWithCastVariant {
  using LoadWithCastPtr = LoadWithCastPtr_List<8>;

  LoadWithCastVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    switch(arity) {
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::LoadWithCast<index>>(iter); break;

      AT_FOR_8_INPUTS(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "make_input_offset_calculator not implemented for ninputs = ", arity);
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
