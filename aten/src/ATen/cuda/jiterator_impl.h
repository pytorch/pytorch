#pragma once
#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/JitLoops.cuh>

#include <string>
#include <variant>
#include <vector>

namespace at::native {


#define AT_FOR_8_CASES(_)  \
  _(1)                      \
  _(2)                      \
  _(3)                      \
  _(4)                      \
  _(5)                      \
  _(6)                      \
  _(7)                      \
  _(8)

#define AT_FOR_8_CASES_WITH_COMMA(_)  \
  _(1)     ,                           \
  _(2)     ,                           \
  _(3)     ,                           \
  _(4)     ,                           \
  _(5)     ,                           \
  _(6)     ,                           \
  _(7)     ,                           \
  _(8)

c10::SmallVector<std::string> get_extra_args_typenames(const c10::SmallVector<at::Scalar>& extra_args) {
  c10::SmallVector<std::string> args_typenames(extra_args.size());
  for (const auto i : c10::irange(extra_args.size())) {
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

template<bool IS_INPUT, int N>
static std::unique_ptr<OffsetCalculator<N>> make_unique_offset_calculator(
          const TensorIteratorBase& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == (IS_INPUT ? iter.ninputs() : iter.noutputs()));

  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    int index = IS_INPUT ? i + iter.noutputs() : i;
    strides[i] = iter.strides(index).data();
    element_sizes[i] = iter.element_size(index);
  }
  return std::make_unique<OffsetCalculator<N>>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <bool IS_INPUT>
struct OffsetCalculatorVariant {
#define DEFINE_CASE(index) std::unique_ptr<OffsetCalculator<index>>
  using OffsetCalculatorTypes = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  OffsetCalculatorVariant(const TensorIteratorBase& iter) {
    int num = IS_INPUT ? iter.ninputs() : iter.noutputs();

    switch(num) {
#define DEFINE_CASE(index)        \
      case index : v = make_unique_offset_calculator<IS_INPUT, index>(iter); break;

      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE
      default:
        TORCH_CHECK(false, "OffsetCalculatorVariant is not implemented for num_tensor = ", num);
    }
  }

  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

 private:
  OffsetCalculatorTypes v;
};

struct ArrayVariant {
// works for up to 8 input + 8 outputs
#define DEFINE_CASE(index) at::detail::Array<char*, index>, at::detail::Array<char*, index+8>
  using ArrayTypes = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  ArrayVariant(const TensorIteratorBase& iter) {
    int ntensors = iter.ntensors();
    switch(ntensors) {
#define DEFINE_CASE(index)                                            \
      case index: array = at::detail::Array<char*, index>{}; break;   \
      case index+8: array = at::detail::Array<char*, index+8>{}; break;

      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "ArrayVariant is not implemented for ntensors = ", ntensors);
    }

    std::visit([&](auto& a) {
      for (auto i = 0; i < ntensors; ++i) {
        a[i] = (char*)iter.data_ptr(i);
      }
    }, array);
  }

  void* data_ptr() {
    return std::visit([](auto & a){ return static_cast<void*>(&a); }, array);
  }

private:
  ArrayTypes array;
};

struct TrivialOffsetCalculatorVariant {
#define DEFINE_CASE(index) TrivialOffsetCalculator<index>
  using TrivialOffsetCalculatorTypes = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  TrivialOffsetCalculatorVariant(int num) {
    switch(num) {
#define DEFINE_CASE(index)      \
      case index: v = TrivialOffsetCalculator<index>(); break;

      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "TrivialOffsetCalculatorVariant is not implemented for num_tensors = ", num);
    }
  }

  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(&v); }, v);
  }

private:
  TrivialOffsetCalculatorTypes v;
};

struct LoadWithCastVariant {
#define DEFINE_CASE(index) std::unique_ptr<memory::LoadWithCast<index>>
  using LoadWithCastPtr = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  LoadWithCastVariant(const TensorIteratorBase& iter) {
    int arity = iter.ninputs();
    switch(arity) {
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::LoadWithCast<index>>(iter); break;

      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "LoadWithCastVariant is not implemented for ninputs = ", arity);
    }
  }

  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

private:
  LoadWithCastPtr v;
};

struct StoreWithCastVariant {
#define DEFINE_CASE(index) std::unique_ptr<memory::StoreWithCast<index>>
  using StoreWithCastPtr = std::variant<
    AT_FOR_8_CASES_WITH_COMMA(DEFINE_CASE)
  >;
#undef DEFINE_CASE

  StoreWithCastVariant(const TensorIteratorBase& iter) {
    int num = iter.noutputs();
    switch(num) {
#define DEFINE_CASE(index)      \
      case index: v = std::make_unique<memory::StoreWithCast<index>>(iter); break;

      AT_FOR_8_CASES(DEFINE_CASE)
#undef DEFINE_CASE

      default:
        TORCH_CHECK(false, "StoreWithCastVariant is not implemented for noutputs = ", num);
    }
  }

  void* data_ptr() {
    return std::visit([](auto & v){ return static_cast<void*>(v.get()); }, v);
  }

private:
  StoreWithCastPtr v;
};

} // namespace at::native


#endif // AT_USE_JITERATOR()
