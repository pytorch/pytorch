#include <ATen/ATen.h>

#define INT_SWITCH_CASE(name, val, ...) \
  case val: {                           \
    constexpr int name = val;           \
    __VA_ARGS__();                      \
    break;                              \
  }

#define DISPATCH_WORLD_SIZES(world_size, ...)      \
  switch (world_size) {                            \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__); \
    default: {                                     \
      constexpr int k_world_size = -1;             \
      __VA_ARGS__();                               \
    }                                              \
  }

#define DISPATCH_WORLD_SIZES_NO_DEFAULT(world_size, ...)                 \
  switch (world_size) {                                                  \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__);                       \
    default: {                                                           \
      TORCH_CHECK(false, "Not implemented for world_size=", world_size); \
    }                                                                    \
  }

#define DISPATCH_ALIGNMENTS_16_8_4(alignment, ...)                     \
  switch (alignment) {                                                 \
    INT_SWITCH_CASE(k_alignment, 16, __VA_ARGS__);                     \
    INT_SWITCH_CASE(k_alignment, 8, __VA_ARGS__);                      \
    INT_SWITCH_CASE(k_alignment, 4, __VA_ARGS__);                      \
    default: {                                                         \
      TORCH_CHECK(false, "Not implemented for alignment=", alignment); \
    }                                                                  \
  }

#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__));
