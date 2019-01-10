#include "torch/csrc/python_headers.h"

#include "THCP.h"

#include "override_macros.h"

#include <system_error>
#include <memory>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THC/THCGenerateAllTypes.h>

