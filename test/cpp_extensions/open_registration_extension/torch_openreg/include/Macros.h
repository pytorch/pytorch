#pragma once

#ifdef _WIN32
#define OPENREG_EXPORT __declspec(dllexport)
#else
#define OPENREG_EXPORT __attribute__((visibility("default")))
#endif
