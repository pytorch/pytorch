// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <caffe2/torch/csrc/jit/backends/xnnpack/executor/xnn_executor.h>
#include <xnnpack.h>
#include <memory>
#include <vector>

namespace torch::jit::xnnpack::delegate {

class XNNCompiler {
 public:
  // Takes Flatbuffer Serialized XNNPack Model and rebuilds the xnn-subgraph
  // returns an executor object that holds the xnn runtime object which we
  // can then use to set inputs and run inference using the xnn graph.
  static void compileModel(
      const void* buffer_pointer,
      size_t num_bytes,
      XNNExecutor* executor);
};

} // namespace torch::jit::xnnpack::delegate
