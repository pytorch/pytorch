#ifndef C10_UTIL_SWITCH_H
#define C10_UTIL_SWITCH_H

#include <c10/macros/Macros.h>

#define C10_EXHAUSTIVE_SWITCH_BEGIN _C10_EXHAUSTIVE_SWITCH_BEGIN_IMP
#define C10_EXHAUSTIVE_SWITCH_END _C10_EXHAUSTIVE_SWITCH_END_IMP

#define C10_FLEXIBLE_SWITCH_BEGIN _C10_FLEXIBLE_SWITCH_BEGIN_IMP
#define C10_FLEXIBLE_SWITCH_END _C10_FLEXIBLE_SWITCH_END_IMP

#define C10_SWITCH_UNEXPECTED_VALUE_ABORT() abort()

#define C10_SWITCH_UNEXPECTED_VALUE_LOG_AND_ABORT(value_type, unexpected_value) \
  ({                                                                        \
    value_type unexpected_value__##__LINE__ = (unexpected_value);           \
    fprintf(                                                                \
        stderr,                                                             \
        "Unexpected " #value_type " value: %ld\n",                          \
        (long)unexpected_value__##__LINE__);                                \
    abort();                                                                \
  })

/*
  SUPPORT MACROS, DO NOT USE DIRECTLY
*/

#if C10_CLANG_HAS_WARNING("-Wcovered-switch-default")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT \
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wcovered-switch-default")
#define _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT \
  C10_CLANG_DIAGNOSTIC_ERROR("-Wcovered-switch-default")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT
#endif

#if C10_CLANG_HAS_WARNING("-Wswitch-enum")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM \
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wswitch-enum")
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM \
  C10_CLANG_DIAGNOSTIC_ERROR("-Wswitch-enum")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM
#endif

#if C10_CLANG_HAS_WARNING("-Wswitch-default")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_DEFAULT \
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wswitch-default")
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT \
  C10_CLANG_DIAGNOSTIC_ERROR("-Wswitch-default")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT
#endif

#define _C10_EXHAUSTIVE_SWITCH_BEGIN_IMP               \
  C10_CLANG_DIAGNOSTIC_PUSH()                          \
  _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT           \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM

#define _C10_EXHAUSTIVE_SWITCH_END_IMP C10_CLANG_DIAGNOSTIC_POP()

#define _C10_FLEXIBLE_SWITCH_BEGIN_IMP               \
  C10_CLANG_DIAGNOSTIC_PUSH()                        \
  _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT         \
  _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM

#define _C10_FLEXIBLE_SWITCH_END_IMP C10_CLANG_DIAGNOSTIC_POP()

#endif // C10_UTIL_SWITCH_H
