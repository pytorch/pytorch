//  Copyright © 2022 Apple Inc.

#define AT_DISPATCH_MPS_TYPES(TYPE, NAME, ...)                                 \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE,                                                                    \
      NAME,                                                                    \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) AT_DISPATCH_CASE(   \
          at::ScalarType::Half,                                                \
          __VA_ARGS__) AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
          AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)                  \
              AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)               \
                  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)         \
                      AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)      \
                          AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))
