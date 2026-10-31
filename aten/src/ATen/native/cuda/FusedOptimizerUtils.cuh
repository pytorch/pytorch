#pragma once

#include <ATen/Dispatch.h>

#define AT_DISPATCH_FUSED_OPTIMIZER_STEP_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE,                                                 \
      NAME,                                                 \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                       \
          at::ScalarType::Float, step_t, __VA_ARGS__)         \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                   \
              at::ScalarType::Double, step_t, __VA_ARGS__))
