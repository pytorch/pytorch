#include "torch/csrc/python_headers.h"
#include <stdarg.h>
#include <string>
#include "THCP.h"

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THC/THCGenerateAllTypes.h>
