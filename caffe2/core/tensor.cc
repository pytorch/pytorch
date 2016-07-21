#include "caffe2/core/tensor.h"
#include "caffe2/core/flags.h"

CAFFE2_DEFINE_bool(
    caffe2_keep_on_shrink, false,
    "If set, keeps memory when a tensor is shrinking its size.");

namespace caffe2 {

namespace detail {

vector<TIndex>& shape(size_t n) {
  static thread_local vector<TIndex> r;
  r.resize(n);
  return r;
}
}
}
