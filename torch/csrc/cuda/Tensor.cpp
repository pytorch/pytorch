#include <Python.h>
#include <structmember.h>

#include <TH/THMath.h>
#include <stdbool.h>
#include <vector>
#include <stack>
#include <tuple>
#include "THCP.h"

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Tensor.cpp"
#include <THC/THCGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/TensorCopy.cpp"
#include <THC/THCGenerateAllTypes.h>

