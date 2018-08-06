#pragma once

#ifdef _WIN32

#if defined(torch_EXPORTS)
#define TORCH_API __declspec(dllexport)
#else
#define TORCH_API __declspec(dllimport)
#endif

#else
#if defined(torch_EXPORTS)
#define TORCH_API
#else
#define TORCH_API
#endif

#endif
