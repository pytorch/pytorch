
copy: fbcode/caffe2/torch/csrc/jit/python/update_graph_executor_opt.cpp
copyrev: 462c83eba55e288fa8ab51376f1b678e0b954739

#include "update_graph_executor_opt.h"

namespace torch {
namespace jit {

thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}
}
}
