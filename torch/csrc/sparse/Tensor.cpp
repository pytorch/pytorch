#include <Python.h>
#include <structmember.h>

#include <stdbool.h>
#include <THS/THS.h>
#include <libshm.h>
#include "THSP.h"
#include "byte_order.h"

#include "torch/csrc/sparse/generic/Tensor.cpp"
#include <THS/THSGenerateAllTypes.h>
