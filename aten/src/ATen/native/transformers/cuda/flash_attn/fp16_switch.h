// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

// modified from static_switch.h
// because MSVC cannot handle std::conditional with constexpr variable

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// FP16_SWITCH(flag, [&] {
///     some_function(...);
/// });
/// ```
#define FP16_SWITCH(COND, ...)                                           \
    [&] {                                                                            \
        if (COND) {                                                                  \
            using elem_type = cutlass::bfloat16_t;   \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            using elem_type =  cutlass::half_t;   \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()