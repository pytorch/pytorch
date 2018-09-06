#ifndef THP_EXPORT_H
#define THP_EXPORT_H

#ifdef __cplusplus
# define THP_EXTERNC extern "C"
#else
# define THP_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef _THP_CORE
#  define THP_API THP_EXTERNC __declspec(dllexport)
#  define THP_CLASS __declspec(dllexport)
# else
#  define THP_API THP_EXTERNC __declspec(dllimport)
#  define THP_CLASS __declspec(dllimport)
# endif
#else
# define THP_API THP_EXTERNC
# define THP_CLASS
#endif

#endif
