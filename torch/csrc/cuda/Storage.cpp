#define __STDC_FORMAT_MACROS

#include "torch/csrc/python_headers.h"
#include <structmember.h>

#include <stdbool.h>
// See Note [TH abstraction violation]
//    - Used to get at allocator from storage
#include <TH/THTensor.hpp>
#include <THC/THCTensor.hpp>
#include "THCP.h"

#include "override_macros.h"
#include "torch/csrc/copy_utils.h"
#include "DynamicTypes.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THC/THCGenerateAllTypes.h>
