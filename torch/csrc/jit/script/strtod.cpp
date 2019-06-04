#include <torch/csrc/jit/script/strtod.h>

#include <stdlib.h>
#include <locale.h>
#include <ATen/core/Macros.h>

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif

#include <errno.h>
#include <locale>

namespace torch {
namespace jit {
namespace script {

#ifdef _WIN32
  C10_EXPORT double strtod_c(const char* str, char** end) {
    /// NOLINTNEXTLINE(hicpp-signed-bitwise)
    static _locale_t loc = _create_locale(LC_ALL, "C");
    return _strtod_l(str, end, loc);
  }
#else
  C10_EXPORT double strtod_c(const char* str, char** end) {
    /// NOLINTNEXTLINE(hicpp-signed-bitwise)
    static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
    return strtod_l(str, end, loc);
  }
#endif


C10_EXPORT float strtof_c(const char *nptr, char **endptr)
{
    return (float) strtod_c(nptr, endptr);
}

}
}
}
