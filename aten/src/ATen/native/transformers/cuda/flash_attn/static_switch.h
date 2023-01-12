// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                           \
    [&] {                                                                            \
        if (COND) {                                                                  \
            constexpr bool CONST_NAME = true;                                        \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            constexpr bool CONST_NAME = false;                                       \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()

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
#define FP16_SWITCH(COND, ...)                                                       \
    [&] {                                                                            \
        if (COND) {                                                                  \
            using elem_type = cutlass::bfloat16_t;                                   \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            using elem_type =  cutlass::half_t;                                      \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()