#include <Python.h>
#include <structmember.h>

#include <TH/THMath.h>
#include <stdbool.h>
#include <vector>
#include <stack>
#include <tuple>
#include "THDP.h"

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Tensor.cpp"
#include <THD/base/THDGenerateAllTypes.h>

//#define THD_GENERIC_FILE "torch/csrc/generic/TensorCopy.cpp"
//#include <THD/base/THDGenerateAllTypes.h>

//#include "undef_macros.h"
//#include "restore_macros.h"

//#include "generic/TensorCopyAsync.cpp"
//#include <THC/THCGenerateAllTypes.h>
