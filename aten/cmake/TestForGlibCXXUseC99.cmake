#.rst:
# TestForGlibCXXUseC99
# --------------------
#
# Check if certain std functions from libcxx are supported and fail if they are
# not. Sometimes _GLIBCXX_USE_C99 macro is not defined and some functions are
# missing.
#
# ::
#
#   SUPPORT_GLIBCXX_USE_C99 - holds result

check_cxx_source_compiles("
#include <cmath>
#include <string>

int main() {
  int a = std::isinf(3.0);
  int b = std::isnan(0.0);
  std::string s = std::to_string(1);

  return 0;
  }" SUPPORT_GLIBCXX_USE_C99)

if(NOT SUPPORT_GLIBCXX_USE_C99)
  message(FATAL_ERROR
          "The C++ compiler does not support required functions. "
          "This is very likely due to a known bug in GCC 5 "
          "(and maybe other versions) on Ubuntu 17.10 and newer. "
          "For more information, see: "
          "https://github.com/pytorch/pytorch/issues/5229"
         )
endif()
