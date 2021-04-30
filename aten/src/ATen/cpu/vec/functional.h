#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/functional.h>
#else
#include <ATen/cpu/vec/vec256/functional.h>
#endif