#include <version>
// clang-format off
#if defined(_LIBCPP_VERSION)
CMAKE-STDLIB-DETECT: libc++
#elif defined(__GLIBCXX__)
CMAKE-STDLIB-DETECT: libstdc++
#else
CMAKE-STDLIB-DETECT: UNKNOWN
#endif
   // clang-format on
