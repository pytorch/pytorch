#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Load.h>
#include <c10/util/TypeCast.h>

namespace c10 {

// Dynamic type casting utils:
// - fetch_and_cast
// - cast_and_store
//
// fetch_and_cast fetch a value with dynamic type specified by a ScalarType
// from a void pointer and cast it to a static type.
//
// cast_and_store casts a static typed value into dynamic type specified
// by a ScalarType, and store it into a void pointer.
//
// NOTE:
//
// Dynamic casting allows us to support type promotion without blowing up
// the combination space: For example, without dynamic cast, in order to
// implement `add_` with type promotion, we would need something like
//
// AT_DISPATCH_ALL_TYPES(output.dtype(),
//    AT_DISPATCH_ALL_TYPES(input1.dtype(),
//       AT_DISPATCH_ALL_TYPES(input2.dtype(),
//           [](arg0_t a, arg1_t b) -> out_t { return a + b; }
//       )
//    )
// )
//
// If we support N dtypes, the above code would generate the a+b kernel for
// all the N * N * N different supported types, the compilation time and
// binary size would become horrible.
//
// Dynamic casting might sounds like a bad idea in terms of performance.
// Especially if you ever do it in a loop, you are going to do a billion tests.
// But in practice it is not as bad as it might look:
//
// - on CPU, this is a branch that always has the same outcome, therefore
//   hopefully the branch predictor could do the job pretty well
// - on GPU, these branches will not diverge, so we could still have the same
//   warp executing the same line of code
// - Most kernels, like `add`, are bandwidth bound, adding a few clock cycles to
//   check an integer does not hurt the performance much because the ALUs would
//   wait for load instructions anyway.
//
// For the discussion and benchmark, refer to:
// - https://github.com/pytorch/pytorch/pull/28343
// - https://github.com/pytorch/pytorch/pull/28344
// - https://github.com/pytorch/pytorch/pull/28345
//

#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

// Fetch a value with dynamic type src_type from ptr, and cast it to static type
// dest_t.
#define FETCH_AND_CAST_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    return c10::convert<dest_t>(c10::load<type>(ptr));

template <typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(
    const ScalarType src_type,
    const void* ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(FETCH_AND_CAST_CASE)
    FETCH_AND_CAST_CASE(uint16_t, UInt16)
    FETCH_AND_CAST_CASE(uint32_t, UInt32)
    FETCH_AND_CAST_CASE(uint64_t, UInt64)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to
// ptr.
#define CAST_AND_STORE_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    *(type*)ptr = c10::convert<type>(value);  \
    return;
template <typename src_t>
C10_HOST_DEVICE inline void cast_and_store(
    const ScalarType dest_type,
    void* ptr,
    src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CAST_AND_STORE_CASE)
    CAST_AND_STORE_CASE(uint16_t, UInt16)
    CAST_AND_STORE_CASE(uint32_t, UInt32)
    CAST_AND_STORE_CASE(uint64_t, UInt64)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}

#define DEFINE_UNCASTABLE(T, scalartype_)                     \
  template <>                                                 \
  C10_HOST_DEVICE inline T fetch_and_cast<T>(                 \
      const ScalarType src_type, const void* ptr) {           \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == src_type);  \
    return c10::load<T>(ptr);                                 \
  }                                                           \
  template <>                                                 \
  C10_HOST_DEVICE inline void cast_and_store<T>(              \
      const ScalarType dest_type, void* ptr, T value) {       \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == dest_type); \
    *(T*)ptr = value;                                         \
  }

AT_FORALL_QINT_TYPES(DEFINE_UNCASTABLE)

#undef FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST

} // namespace c10
