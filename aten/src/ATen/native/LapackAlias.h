#pragma once

// macOS 13.3+ ships BLAS/LAPACK interfaces aligned with Reference LAPACK 3.9.1 next to the deprecated
// LAPACK 3.2.1 ones, under `$NEWLAPACK`-suffixed symbol names, see
// Accelerate.framework/Frameworks/vecLib.framework/Headers/lapack_version.h
// AT_LAPACK_ALIAS makes PyTorch's own BLAS/LAPACK prototypes bind to the new ones.
#if defined(__APPLE__) && defined(ACCELERATE_NEW_LAPACK)
#define AT_LAPACK_ALIAS(name) __asm("_" #name "$NEWLAPACK")
#else
#define AT_LAPACK_ALIAS(name)
#endif
