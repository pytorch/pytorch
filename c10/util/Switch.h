// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef C10_UTIL_SWITCH_H
#define C10_UTIL_SWITCH_H

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

#if defined(__has_warning) && (defined(__GNUC__) || defined(__clang__))
#define _C10_LANG_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define _C10_LANG_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#if __has_warning("-Wcovered-switch-default")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT \
  _Pragma("GCC diagnostic ignored \"-Wcovered-switch-default\"")
#define _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT \
  _Pragma("GCC diagnostic error \"-Wcovered-switch-default\"")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT
#endif
#if __has_warning("-Wswitch-enum")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM \
  _Pragma("GCC diagnostic ignored \"-Wswitch-enum\"")
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM \
  _Pragma("GCC diagnostic error \"-Wswitch-enum\"")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM
#endif
#if __has_warning("-Wswitch-default")
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_DEFAULT \
  _Pragma("GCC diagnostic ignored \"-Wswitch-default\"")
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT \
  _Pragma("GCC diagnostic error \"-Wswitch-default\"")
#else
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT
#endif
#else
#define _C10_LANG_DIAGNOSTIC_PUSH
#define _C10_LANG_DIAGNOSTIC_POP
#define _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM
#define _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_DEFAULT
#define _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT
#endif

#define _C10_EXHAUSTIVE_SWITCH_BEGIN_IMP               \
  _C10_LANG_DIAGNOSTIC_PUSH                            \
  _C10_LANG_DIAGNOSTIC_IGNORED_WCOVERED_SWITCH_DEFAULT \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT           \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_ENUM

#define _C10_EXHAUSTIVE_SWITCH_END_IMP _C10_LANG_DIAGNOSTIC_POP

#define _C10_FLEXIBLE_SWITCH_BEGIN_IMP               \
  _C10_LANG_DIAGNOSTIC_PUSH                          \
  _C10_LANG_DIAGNOSTIC_ERROR_WCOVERED_SWITCH_DEFAULT \
  _C10_LANG_DIAGNOSTIC_ERROR_WSWITCH_DEFAULT         \
  _C10_LANG_DIAGNOSTIC_IGNORED_WSWITCH_ENUM

#define _C10_FLEXIBLE_SWITCH_END_IMP _C10_LANG_DIAGNOSTIC_POP

#endif // C10_UTIL_SWITCH_H
