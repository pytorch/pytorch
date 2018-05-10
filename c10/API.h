#pragma once

#ifdef _WIN32
# ifdef c10_EXPORTS
#  define C10_API __declspec(dllexport)
# else
#  define C10_API __declspec(dllimport)
# endif
#else
# define C10_API
#endif
