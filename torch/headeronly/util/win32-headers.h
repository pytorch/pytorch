#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOKERNEL
#define NOKERNEL
#endif
#ifndef NOUSER
#define NOUSER
#endif
#ifndef NOSERVICE
#define NOSERVICE
#endif
#ifndef NOSOUND
#define NOSOUND
#endif
#ifndef NOMCX
#define NOMCX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#ifndef NOMSG
#define NOMSG
#endif
#ifndef NOMB
#define NOMB
#endif
#ifndef NOCLIPBOARD
#define NOCLIPBOARD
#endif

// dbghelp seems to require windows.h.
// clang-format off
#include <windows.h>
#include <dbghelp.h>
// clang-format on

#undef VOID
#undef DELETE
#undef IN
#undef THIS
#undef CONST
#undef NAN
#undef UNKNOWN
#undef NONE
#undef ANY
#undef IGNORE
#undef STRICT
#undef GetObject
#undef CreateSemaphore
#undef Yield
#undef RotateRight32
#undef RotateLeft32
#undef RotateRight64
#undef RotateLeft64
