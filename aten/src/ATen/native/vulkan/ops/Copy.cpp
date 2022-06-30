#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor& copy_(Tensor& self, const Tensor& src) {
  api::Context* const context = api::context();

  api::Command::Pool& command_pool = context->command().pool;
  {
    // X -> Vulkan
    if (at::kVulkan == self.device().type()) {
      vTensor& v_self = convert(self);

      // Vulkan -> Vulkan
      if (at::kVulkan == src.device().type()) {
       api::Command::Buffer& command_buffer = command_pool.stream();
       {
          api::OpProfiler profiler(command_buffer, context->querypool(), "copy_");

          command_buffer.copy(
              // - Read-only access is implied on const tensors.  Memory barriers
              //   are automatically inserted if a RAW hazard is detected.
              // - Recording any potential pending sync operations into the same
              //   command buffer prevents an expensive queue submission.
              convert(src).buffer(
                  command_buffer,
                  vTensor::Stage::Transfer),
              // - Write-only access never triggers a sync as the contents will be
              //   overwritten regardless.  Having said that, appropriate barriers
              //   are inserted automatically if WAR or WAW hazards are detected.
              // - Recording pending sync operations into the same command buffer
              //   prevents an expensive queue submission.
              v_self.buffer(
                  command_buffer,
                  vTensor::Stage::Transfer,
                  vTensor::Access::Write));
        }
        command_pool.submit(context->gpu().queue, command_buffer);
      }
      // CPU -> Vulkan
      else {
        api::Command::Buffer& command_buffer = command_pool.stream(); // Don't collect the timestamp since the command buffer doesn't record anything
        const Tensor cpu_src = src.device().is_cpu() ? src : src.cpu();

        {
          api::MemoryMap mapping(
              v_self.host_buffer(command_buffer, vTensor::Access::Write),
              api::MemoryAccessType::WRITE);

          float* data_ptr = mapping.template data<float>();

          memcpy(
            data_ptr,
            cpu_src.contiguous().data_ptr<float>(),
            std::min(src.nbytes(), self.nbytes()));
        }
      }
    }
    // Vulkan -> X
    else if (at::kVulkan == src.device().type()) {
      api::Command::Buffer& command_buffer = command_pool.stream(); // Don't collect the timestamp since the command buffer doesn't record anything
      vTensor& v_src = convert(src);

      // Vulkan -> CPU
      if (self.device().is_cpu()) {
        {
          api::MemoryMap mapping(
            v_src.host_buffer(command_buffer, vTensor::Access::Read),
            api::MemoryAccessType::READ);

          v_src.wait_for_fence();
          mapping.invalidate();

          float* data_ptr = mapping.template data<float>();

          memcpy(
              self.data_ptr<float>(),
              data_ptr,
              std::min(src.nbytes(), self.nbytes()));
        }
      }
      else {
        TORCH_CHECK(false, "Unsupported!");
      }

      //
      // WARNING
      //

      // This is not great.  We almost never want to flush the GPU pipeline as
      // that has far reaching consequences, especially if PyTorch is not the only
      // process accessing the GPU.  If we have done our job properly, above
      // synchronization mechanisms should be enough to ensure correctness at a more
      // modest cost, as there is no need to flush the entirety of jobs in flight
      // if one is only interested on waiting on computation affecting one single
      // tensor to finish.
      //
      // Having said that, we still do need to release all pool resources at one
      // point per inference run or we will run out of memory otherwise. There is
      // no perfect answer to this problem that checks all boxes, which leaves us
      // with one of several design decisions:
      //
      // 1) Use graph mode to gain an understanding of the computation graph,
      //    itself allowing us to place pool purges intelligently.  Best option
      //    for performance and memory consumption.  Not without its downsides if
      //    flexibility is a top priority.
      // 2) If on eager mode, and hence are seeing operations one at a time, expose
      //    this release of resources to the user as a Python / C++ function.  This
      //    makes for suboptimal user experience but is efficient in terms of
      //    performance.
      // 3) If on eager mode, and interested in keeping this bookkeeping transparent
      //    to the user, release all resources somewhere ... like here.  This is
      //    not ideal since it requires a pipeline flush to make sure these objects
      //    are not already in use by a workload in flight.  Cannot do much better
      //    within the constraints of this approach.  Good for user experience,
      //    suboptimal for performance.
      // 4) If on eager mode, and interested in keeping this bookkeeping transparent
      //    to the user, and performance does not matter, make CPU and GPU run in
      //    lockstep.  Obviously this is just bad.  Mentioned for the sake of
      //    completeness.

      context->flush();
    }
    else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Invalid code path taken! Either the source or the destination tensor "
          "was expected to be Vulkan a tensor!  Incorrect dispatch?");
    }
  }
  // No queue submission here.  All queue submissions must have been handled
  // above either explicitly or as a result of calling tensor.host().

  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
