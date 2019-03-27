#pragma once

#include <c10/macros/Export.h>

#ifdef _WIN32
#define TORCH_API CAFFE2_API
#elif defined(__GNUC__)
#if defined(torch_EXPORTS)
#define TORCH_API __attribute__((__visibility__("default")))
#else
#define TORCH_API
#endif
#else
#define TORCH_API
#endif
