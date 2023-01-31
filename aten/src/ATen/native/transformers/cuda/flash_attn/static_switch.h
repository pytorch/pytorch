// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
// and https://github.com/facebookresearch/xformers/blob/main/xformers/csrc/attention/cuda/fmha/gemm_kernel_utils.h#L8

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, ([&] {
///     some_function<BoolConst>(...);
/// }));
/// ```
/// We need "({" and "})" to make sure that the code is a single argument being passed to the macro.
#define BOOL_SWITCH(COND, CONST_NAME, F)       \
    {                                          \
        if (COND) {                            \
            constexpr bool CONST_NAME = true;  \
            F();                               \
        } else {                               \
            constexpr bool CONST_NAME = false; \
            F();                               \
        }                                      \
    }

// modified from BOOL_SWITCH
// because MSVC cannot handle std::conditional with constexpr variable
#define FP16_SWITCH(COND, F)                 \
    {                                        \
        if (COND) {                          \
            using elem_type = __nv_bfloat16; \
            F();                             \
        } else {                             \
            using elem_type = __half;        \
            F();                             \
        }                                    \
    }
