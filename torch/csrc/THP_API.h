#ifndef THP_API_H
#define THP_API_H

#ifdef _THP_CORE
#error Using the THP API header, but _THP_CORE is defined! This macro should \
    be defined only when compiling the core torch package.
#endif

#ifdef USE_CUDA
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/undef_macros.h>
#endif

#include <torch/csrc/THP.h>

#endif
