#ifndef THP_API_H
#define THP_API_H

#ifdef _THP_CORE
#error Using the THP API header, but _THP_CORE is defined! This macro should \
    be defined only when compiling the core torch package.
#endif

#ifdef WITH_CUDA
#include "cuda/THCP.h"
#include "cuda/undef_macros.h"
#endif

#include "THP.h"

#endif
