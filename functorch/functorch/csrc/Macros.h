#pragma once

// FUNCTORCH_BUILD_MAIN_LIB is set in setup.py.
// We don't really need to use C10_IMPORT because no C++ project relies on
// functorch. But leaving it here for future-proofing.
#ifdef FUNCTORCH_BUILD_MAIN_LIB
#define FUNCTORCH_API C10_EXPORT
#else
#define FUNCTORCH_API C10_IMPORT
#endif
