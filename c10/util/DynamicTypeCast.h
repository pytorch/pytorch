#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>


namespace c10 {

// Fetch a value with type src_type from ptr, and cast it to dest_t.
#define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast<dest_t>(*(const type *)ptr);
template<typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, FETCH_AND_CAST_CASE)
    default:;
  }
#ifdef C10_HOST_DEVICE
  assert(false);
#else
  TORCH_CHECK(false, "Unexpected scalar type");
#endif
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with type src_t into dest_type, and store it to ptr.
#define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: *(type *)ptr = static_cast<type>(value); return;
template<typename src_t>
C10_HOST_DEVICE inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, CAST_AND_STORE_CASE)
    default:;
  }
#ifdef C10_HOST_DEVICE
  assert(false);
#else
  TORCH_CHECK(false, "Unexpected scalar type");
#endif
}

// #undef FETCH_AND_CAST_CASE
// #undef CAST_AND_STORE_CASE

}  // namespace c10