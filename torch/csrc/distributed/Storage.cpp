#include <Python.h>
#include <structmember.h>

#include <stdbool.h>
#include "THDP.h"

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THD/base/THDGenerateAllTypes.h>

//#define THD_GENERIC_FILE "torch/csrc/generic/StorageCopy.cpp"
//#include <THD/THDGenerateAllTypes.h>

