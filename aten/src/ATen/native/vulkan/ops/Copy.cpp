#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor& copy_(Tensor& self, const Tensor& src) {
  // X -> Vulkan
  if (at::kVulkan == self.device().type()) {
    vTensor& v_self = convert(self);

    // CPU -> Vulkan
    if (at::kCPU == src.device().type()) {
      // Requesting write-only host access to the tensor never triggers a sync
      // as the contents will be overwritten regardless.  Having said that,
      // appropriate barriers are inserted automatically if WAR or WAW hazards
      // are detected.  Examples of such scenario for instance are if any of
      // these async operations are on going in the background on 'self':
      //  - On discrete systems:
      //      * buffer-to-staging transfers
      //      * staging-to-buffer transfers
      // -  On UMA buffer is an alias for staging and accessible both on host
      //    and device.  Consequently:
      //      * buffer-to-image NHWC -> NC4HW packing
      //      * image-to-buffer NC4HW -> NHWC unpacking

      using Future = vTensor::Future<void, vTensor::Access::Write>;
      Future v_self_future = v_self.host<void, vTensor::Access::Write>();

      // This wait() will be a no-op if no hazards are detected, including the
      // obvious, yet important, special case of 'self' being an empty tensor.

      Future::Payload v_self_payload = v_self_future.wait();

      memcpy(
        v_self_payload.get(),
        src.contiguous().data_ptr<float>(),
        std::min(src.nbytes(), self.nbytes()));
    }
    // Vulkan -> Vulkan
    else if (at::kVulkan == src.device().type()) {
      api::Command::Buffer command_buffer = api::context()->command().pool.allocate();
      command_buffer.begin();

      command_buffer.copy(
          // - Read-only access is implied on const tensors.  Memory barriers
          //   are automatically inserted if a RAW hazard is detected.
          // - Recording any potential pending sync operations into the same
          //   command buffer prevents an expensive queue submission.
          convert(src).buffer(command_buffer),
          // - Write-only access never triggers a sync as the contents will be
          //   overwritten regardless.  Having said that, appropriate barriers
          //   are inserted automatically if WAR or WAW hazards are detected.
          // - Recording pending sync operations into the same command buffer
          //   prevents an expensive queue submission.
          v_self.buffer(command_buffer, vTensor::Access::Write));

      command_buffer.end();
      command_buffer.submit(api::context()->gpu().queue);
    }
    else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported!");
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    const vTensor& v_src = convert(src);

    {
      // Similar notes as above applies, with the additional consideration of
      // potential syncs on read accesses.  Namely,
      // - on discrete systems, if the (staging, buffer, image) trio, or
      // - on UMA, if the (buffer, image) duo
      // have gone out of sync as a result of one processor writing to one
      // resource which is then either accessed as an another resource type on
      // the same or another processor.  Same considerations regarding hazard
      // avoidance as above applies.

      using Future = vTensor::Future<const void, vTensor::Access::Read>;
      const Future v_src_future = v_src.host<const void>();

      // Vulkan -> CPU
      if (at::kCPU == self.device().type()) {
        // This wait() is a no-op if data is not out of sync.  More often than
        // not though, waits here are expected as the GPU catches up with
        // compute submitted from CPU.

        const Future::Payload v_src_payload = v_src_future.wait();

        memcpy(
          self.data_ptr<float>(),
          v_src_payload.get(),
          std::min(src.nbytes(), self.nbytes()));
      }
      else {
        TORCH_INTERNAL_ASSERT(false, "Unsupported!");
      }
    }

    //
    // WARNING
    //

    // This is bad practice.  We almost never want to flush the GPU pipeline as
    // that has far reaching consequences, especially if PyTorch is not the only
    // process accessing the GPU.  If we have done our job properly, above
    // synchronization mechanisms should be enough to ensure correctness at a more
    // modest cost, as there is no need to flush the entirety of jobs in flight
    // if one is only interested on waiting on computation affecting one single
    // tensor to finish.
    //
    // Having said that, we still do need to release all pool resources at one
    // point per inference run or otherwise we will run out of memory. There is
    // no perfect answer to this problem that checks all boxes, which leavs us
    // with one of several design decisions:
    //
    // 1) Use graph mode to gain an understanding of the computation graph,
    //    itself allowing us to place pool purges intelligently.  Best option
    //    for performance and memory consumption.  Not without its downsides if
    //    flexibility is a top priority.
    // 2) If on eager mode, and hence are seeing operations one at a time, expose
    //    this release of resources to the user as a Python / C++ function.
    // 3) If on eager mode, and interested in keeping this bookkeeping transparent
    //    to the user, release all resources somewhere ... like here.  This is
    //    not ideal since it requires a pipeline flush to make sure these objects
    //    are not already in use but cannot do much better within the constraints
    //    of this approach.
    // 4) If on eager mode, and interested in keeping this bookkeeping transparent
    //    to the user, and performance does not matter, make CPU and GPU run in
    //    lockstep.  Obviously this is just bad.  Mentioned for the sake of
    //    completeness.

    api::context()->flush();
  }
  else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported!");
  }

  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
