#pragma once

#ifdef _WIN32
# if defined(C10_BUILD_MAIN_LIB)
#  define C10_API __declspec(dllexport)
# else
#  define C10_API __declspec(dllimport)
# endif
#else
# define C10_API
#endif
