//  Copyright © 2022 Apple Inc.

#define AT_DISPATCH_MPS_TYPES(TYPE, NAME, ...)                                \
  [&] {                                                                       \
    const auto& the_type = TYPE;                                              \
    at::ScalarType _st = ::detail::scalar_type(the_type);                     \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                  \
    switch (_st) {                                                            \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                                \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        \
    }                                                                         \
  }()
