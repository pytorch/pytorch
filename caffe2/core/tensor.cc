#include "caffe2/core/flags.h"

CAFFE2_DEFINE_bool(
    caffe2_keep_on_shrink, false,
    "If set, keeps memory when a tensor is shrinking its size.");
