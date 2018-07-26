#ifndef CAFFE2_OPERATORS_CONV_OP_SHARED_H_
#define CAFFE2_OPERATORS_CONV_OP_SHARED_H_

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

/**
 * Creates a mutex and shared buffer in the workspace.
 * Not thread-safe, must be called from the constructor.
 */
template <typename Context>
void createSharedBuffer(Workspace* ws);

/**
 * Thread-safe, can be invoked from RunOnDevice() to serialize
 * access to shared buffer.
 */
template <typename Context>
void runWithSharedBuffer(Workspace* ws, std::function<void(Tensor* buffer)> f);
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_SHARED_H_
