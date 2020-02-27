
copy: fbcode/caffe2/torch/csrc/jit/python/update_graph_executor_opt.h
copyrev: 1e7df1bbe334b159b6c34d0d65606a721eb2b4cd

#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
namespace torch {
namespace jit{
TORCH_API void setGraphExecutorOptimize(bool o);
TORCH_API bool getGraphExecutorOptimize();
}
}
