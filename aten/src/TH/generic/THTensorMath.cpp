#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>
#include <ATen/NamedTensorUtils.h>

// HEY YOU!
//
// Looking for a function which used to be in THTensorMath.cpp, but
// can't find it anymore?  Check THTensorMoreMath.cpp and
// THTensorEvenMoreMath.cpp.  These source files have been split up
// because they were getting too big (a whopping 4669 lines at time
// of writing) and causing MSVC to run out of memory.  Did you come
// here because you saw:
//
//    fatal error C1002: compiler is out of heap space in pass 2
//
// Try splitting up the file some more.
//
// At some point, we should reorganize these files in a way that makes
// sense (rather than just having cut the file down the middle, which is
// what I did when I split these up originally).

#endif /* TH_GENERIC_FILE */
