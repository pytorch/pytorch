#pragma once

#include <ATen/cpu/vec/vec256/functional_base.h>
#if !defined(__VSX__)  || !defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec/vec256/functional_bfloat16.h>
#endif
