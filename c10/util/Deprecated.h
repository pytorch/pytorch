#pragma once

// Largely from https://stackoverflow.com/questions/295120/c-mark-as-deprecated

#if defined(__cplusplus) && __cplusplus > 201402L
#define C10_DEPRECATED(function) [[deprecated]] function
#else
#if defined(__GNUC__)
#define C10_DEPRECATED(function) __attribute__((deprecated)) function
#elif defined(_MSC_VER)
#define C10_DEPRECATED(function) __declspec(deprecated) function
#else
#warning "You need to implement C10_DEPRECATED for this compiler"
#define C10_DEPRECATED(function) function
#endif // defined(__GNUC__)
#endif // defined(__cplusplus) && __cplusplus > 201402L
