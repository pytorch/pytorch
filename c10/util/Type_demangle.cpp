#include <c10/util/Type.h>

#include <cstdlib>
#include <functional>
#include <memory>

#if HAS_DEMANGLE

#include <cxxabi.h>

namespace c10 {

std::string demangle(const char* name) {
  int status = -1;

  // This function will demangle the mangled function name into a more human
  // readable format, e.g. _Z1gv -> g().
  // More information:
  // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
  // NOTE: `__cxa_demangle` returns a malloc'd string that we have to free
  // ourselves.
  std::unique_ptr<char, std::function<void(char*)>> demangled(
      abi::__cxa_demangle(
          name,
          /*__output_buffer=*/nullptr,
          /*__length=*/nullptr,
          &status),
      /*deleter=*/free);

  // Demangling may fail, for example when the name does not follow the
  // standard C++ (Itanium ABI) mangling scheme. This is the case for `main`
  // or `clone` for example, so the mangled name is a fine default.
  if (status == 0) {
    return demangled.get();
  } else {
    return name;
  }
}

} // namespace c10

#endif
