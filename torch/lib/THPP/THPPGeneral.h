#pragma once

#ifdef _WIN32
# ifdef THPP_EXPORTS
#  define THPP_API THC_EXTERNC __declspec(dllexport)
#  define THPP_CLASS __declspec(dllexport)
# else
#  define THPP_API THC_EXTERNC __declspec(dllimport)
#  define THPP_CLASS __declspec(dllimport)
# endif
#else
# define THPP_API extern
# define THPP_CLASS
#endif
