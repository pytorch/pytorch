#pragma once

#ifdef _WIN32
# ifdef ATen_EXPORTS
#  define ATen_API extern "C" __declspec(dllexport)
#  define ATen_CLASS __declspec(dllexport)
# else
#  define ATen_API extern "C" __declspec(dllimport)
#  define ATen_CLASS __declspec(dllimport)
# endif
#else
# define ATen_API extern
# define ATen_CLASS
#endif
