#pragma once

#ifdef _WIN32
# ifdef ATen_EXPORTS
#  define AT_API __declspec(dllexport)
# else
#  define AT_API __declspec(dllimport)
# endif
#else
# define AT_API
#endif
