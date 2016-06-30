#ifndef THCP_UTILS_H
#define THCP_UTILS_H

#include "THCP.h"

#include "override_macros.h"

#define THCPUtils_(NAME) TH_CONCAT_4(THCP,Real,Utils_,NAME)

#define THC_GENERIC_FILE "torch/csrc/generic/utils.h"
#include <THC/THCGenerateAllTypes.h>

#endif
