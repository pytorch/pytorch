#include <Python.h>
#include <structmember.h>

#include <stdbool.h>
#include "THCP.h"

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THC/THCGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/StorageCopy.cpp"
#include <THC/THCGenerateAllTypes.h>
