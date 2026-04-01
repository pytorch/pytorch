// nvapi-include.h
#pragma once

// A helper that makes the NVAPI available across targets

#ifdef GFX_NVAPI
// On windows if we include NVAPI, we must include windows.h first

#ifdef _WIN32
#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("NOMINMAX")
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#define NOMINMAX
#include <windows.h>
#pragma pop_macro("NOMINMAX")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")
#endif

#include <nvShaderExtnEnums.h>
#include <nvapi.h>

#endif
