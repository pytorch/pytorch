#pragma once
#if defined(__GNUC__) && __GNUC__ == 10 && __GNUC_MINOR__ <= 2 && \
    defined(__ARM_FEATURE_SVE)
// Workaround for https: //gcc.gnu.org/bugzilla/show_bug.cgi?id=117161
#pragma GCC optimize("no-tree-vectorize")
#endif

#include <torch/headeronly/cpu/vec/vec_base.h>

// additional headers for more operations that depend on vec_base
#include <ATen/cpu/vec/vec_convert.h>
#include <ATen/cpu/vec/vec_mask.h>
#include <ATen/cpu/vec/vec_n.h>
