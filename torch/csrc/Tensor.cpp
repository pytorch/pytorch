#include <Python.h>
#include <structmember.h>

#define THP_HOST_HALF

#include <stdbool.h>
#include <vector>
#include <stack>
#include <tuple>
#include <TH/THMath.h>

#include "THP.h"
#include "copy_utils.h"
#include "DynamicTypes.h"

#include "generic/Tensor.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/Tensor.cpp"
#include <TH/THGenerateHalfType.h>
