/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_ring.h"

#include "gloo/cuda_collectives_device.h"
#include "gloo/cuda_collectives_host.h"
#include "gloo/cuda_private.h"

namespace gloo {

template <typename T, typename W>
CudaAllreduceRing<T, W>::CudaAllreduceRing(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum),
      leftPair_(this->getLeftPair()),
      rightPair_(this->getRightPair()) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Workspace specific initialization (see below)
  init();

  auto slot = this->context_->nextSlot();

  // Buffer to send to (rank+1).
  sendDataBuf_ = rightPair_->createSendBuffer(slot, *outbox_, bytes_);

  // Buffer that (rank-1) writes to.
  recvDataBuf_ = leftPair_->createRecvBuffer(slot, *inbox_, bytes_);

  // Dummy buffers for localized barrier.
  // Before sending to the right, we only need to know that the node
  // on the right is done using the inbox that's about to be written
  // into. No need for a global barrier.
  auto notificationSlot = this->context_->nextSlot();
  sendNotificationBuf_ =
    leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  recvNotificationBuf_ =
    rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
}

template <typename T, typename W>
void CudaAllreduceRing<T, W>::run() {
  CudaDeviceGuard guard;
  CudaStream& stream = streams_[0];

  if (localReduceOp_) {
    localReduceOp_->run();
  }

  // Initialize outbox with locally reduced values
  stream.copyAsync(outbox_, scratch_);
  stream.wait();

  int numRounds = this->contextSize_ - 1;
  for (int round = 0; round < numRounds; round++) {
    // Initiate write to inbox of node on the right
    sendDataBuf_->send();

    // Wait for inbox write from node on the left
    recvDataBuf_->waitRecv();

    // Reduce
    fn_->call(scratch_, inbox_, count_, stream);
    stream.wait();

    // Wait for outbox write to complete
    sendDataBuf_->waitSend();

    // Prepare for next round if necessary
    if (round < (numRounds - 1)) {
      stream.copyAsync(outbox_, inbox_);
      stream.wait();
    }

    // Send notification to node on the left that
    // this node is ready for an inbox write.
    sendNotificationBuf_->send();

    // Wait for notification from node on the right
    recvNotificationBuf_->waitRecv();
  }

  // Asynchronously copy result buffer to all device buffers
  if (localBroadcastOp_) {
    localBroadcastOp_->runAsync();
    if (synchronizeDeviceOutputs_) {
      localBroadcastOp_->wait();
    }
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceRing<T, W>::init(
    typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                            typename U::Pointer>::type*) {
  // Since reduction is executed on the CPU, the scratch space
  // where they are accumulated is a new host side buffer.
  scratch_ = W::Pointer::alloc(count_);

  // Execute local reduction and broadcast from host.
  // If devicePtrs_.size() == 1 these functions construct an op that
  // executes a memcpy such that scratch_ always holds the result.
  localReduceOp_ =
    cudaHostReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
  localBroadcastOp_ =
    cudaHostBroadcast(streams_, devicePtrs_, scratch_, 0, count_);

  inbox_ = W::Pointer::alloc(count_);
  outbox_ = W::Pointer::alloc(count_);
}

template <typename T, typename W>
template <typename U>
void CudaAllreduceRing<T, W>::init(
    typename std::enable_if<std::is_same<U, CudaDeviceWorkspace<T> >::value,
                            typename U::Pointer>::type*) {
  // Since reduction is executed on the GPU, the scratch space
  // can use an existing input buffer to accumulate.
  auto& ptr = devicePtrs_[0];
  auto count = ptr.getCount();
  scratch_ = CudaDevicePointer<T>::create(*ptr, count);

  // Run local reduction and broadcast on device.
  // When running with a device workspace we intend to never leave the device.
  if (devicePtrs_.size() > 1) {
    localReduceOp_ =
      cudaDeviceReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
    localBroadcastOp_ =
      cudaDeviceBroadcast(streams_, devicePtrs_, scratch_, 0, count_);
  }

  // Inbox/outbox must be colocated with scratch buffer to avoid
  // cross device copies while accumulating the reduction.
  {
    CudaDeviceScope scope(scratch_.getDeviceID());
    inbox_ = W::Pointer::alloc(count);
    outbox_ = W::Pointer::alloc(count);
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceRing<T, CudaHostWorkspace<T> >;             \
template class CudaAllreduceRing<T, CudaDeviceWorkspace<T> >;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
