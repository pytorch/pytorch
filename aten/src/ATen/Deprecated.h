#pragma once

// Largely from https://stackoverflow.com/questions/295120/c-mark-as-deprecated

#if defined(__cplusplus) && __cplusplus > 201402L
#define AT_DEPRECATED(function) [[deprecated]] function
#else
#if defined(__GNUC__)
#define AT_DEPRECATED(function) __attribute__((deprecated)) function
#elif defined(_MSC_VER)
#define AT_DEPRECATED(function) __declspec(deprecated) function
#else
#warning "You need to implement AT_DEPRECATED for this compiler"
#define AT_DEPRECATED(function) function
#endif // defined(__GNUC__)
#endif // defined(__cplusplus) && __cplusplus > 201402L
