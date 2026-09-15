#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#include <atomic>

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
#ifdef __cpp_lib_atomic_ref
  std::atomic_ref<float>(*dst).fetch_add(fvalue);
#else
  __atomic_fetch_add(dst, fvalue, __ATOMIC_SEQ_CST);
#endif
}

#endif // ATOMIC_ADD_FLOAT
