#include <Python.h>
#include <structmember.h>

#define THP_HOST_HALF

#include <stdbool.h>
#include <vector>
#include <stack>
#include <tuple>
#include <TH/THMath.h>

#include "torch/csrc/distributed/THDP.h"
#include "torch/csrc/copy_utils.h"
#include "torch/csrc/DynamicTypes.h"

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Tensor.cpp"
#include <THD/base/THDGenerateAllTypes.h>

//#define THD_GENERIC_FILE "torch/csrc/generic/TensorCopy.cpp"
//#include <THD/base/THDGenerateAllTypes.h>

