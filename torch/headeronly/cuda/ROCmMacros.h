#pragma once

#if defined(USE_ROCM)

#if !__has_builtin(__builtin_amdgcn_processor_is)
  #if defined(__amdgcn_processor__)
    // Device pass: __amdgcn_processor__ is available
    #define __builtin_amdgcn_processor_is(x) (__builtin_strcmp(x, __amdgcn_processor__) == 0)
  #else
    // Host pass: define a no-op fallback so the macro always exists
    #define __builtin_amdgcn_processor_is(x) false
  #endif // defined(__amdgcn_processor__)
#endif // !__has_builtin(__builtin_amdgcn_processor_is)

#if !__has_builtin(__builtin_amdgcn_is_invocable)
  #define __builtin_amdgcn_is_invocable(x) (__has_builtin(x))
#endif

#endif // defined(USE_ROCM)
