// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <caffe2/torch/csrc/jit/backends/xnnpack/executor/xnn_executor.h>
#include <xnnpack.h>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNCompiler {
 public:
  // Takes Flatbuffer Serialized XNNPack Model and rebuilds the xnn-subgraph
  // returns an executor object that holds the xnn runtime object which we
  // can then use to set inputs and run inference using the xnn graph.
  static XNNExecutor compileModel(std::string ser_model);
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
