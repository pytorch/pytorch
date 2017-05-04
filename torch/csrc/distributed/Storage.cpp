#include <Python.h>
#include <structmember.h>

#define THP_HOST_HALF

#include <TH/TH.h>
#include <libshm.h>
#include "THDP.h"
#include "copy_utils.h"

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THD/base/THDGenerateAllTypes.h>

//#define THD_GENERIC_FILE "torch/csrc/generic/StorageCopy.cpp"
//#include <THD/THDGenerateAllTypes.h>

