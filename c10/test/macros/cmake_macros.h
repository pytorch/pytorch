#include <c10/macros/Macros.h>

namespace c10 {

constexpr bool build_shared_libs =
#if defined(C10_BUILD_SHARED_LIBS)
    true
#else
    false
#endif
    ;

constexpr bool use_glog =
#if defined(C10_USE_GLOG)
    true
#else
    false
#endif
    ;

constexpr bool use_gflags =
#if defined(C10_USE_GFLAGS)
    true
#else
    false
#endif
    ;

constexpr bool use_numa =
#if defined(C10_USE_NUMA)
    true
#else
    false
#endif
    ;

constexpr bool use_msvc_static_runtime =
#if defined(C10_USE_MSVC_STATIC_RUNTIME)
    true
#else
    false
#endif
    ;

} // namespace c10
