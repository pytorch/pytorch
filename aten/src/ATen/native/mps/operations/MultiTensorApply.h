#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

static_assert(sizeof(bool) == 1);

namespace at::native::mps {

static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kmaxThreadGroups = 32;
static constexpr int64_t kmaxTensors = 32;

struct MetadataArguments { // the size of this struct must be less than 4 kilobytes
  uint64_t numels[kmaxTensors];
  uint64_t threadgroup_to_tensor[kmaxThreadGroups];
  uint64_t threadgroup_to_chunk[kmaxThreadGroups];
};

struct FusedAdamEncodingFunctor {
  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const double lr,
                  const double beta1,
                  const double beta2,
                  const double weight_decay,
                  const double eps,
                  const bool maximize) const {
    mtl_setArgs(
        computeEncoder, tensorArgumentBuffer, metadata_arguments, lr, beta1, beta2, weight_decay, eps, maximize);
  }

  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const at::Tensor& lr,
                  const double beta1,
                  const double beta2,
                  const double weight_decay,
                  const double eps,
                  const bool maximize) const {
    mtl_setArgs(
        computeEncoder, tensorArgumentBuffer, metadata_arguments, lr, beta1, beta2, weight_decay, eps, maximize);
  }
};

template <bool momentum>
struct FusedSgdEncodingFunctor {};

template <>
struct FusedSgdEncodingFunctor<true> {
  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const double weight_decay,
                  const double momentum,
                  const double lr,
                  const double dampening,
                  const bool nesterov,
                  const bool maximize,
                  const bool is_first_step) const {
    mtl_setArgs(computeEncoder,
                tensorArgumentBuffer,
                metadata_arguments,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov,
                maximize,
                is_first_step);
  }

  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const double weight_decay,
                  const double momentum,
                  const at::Tensor& lr,
                  const double dampening,
                  const bool nesterov,
                  const bool maximize,
                  const bool is_first_step) const {
    mtl_setArgs(computeEncoder,
                tensorArgumentBuffer,
                metadata_arguments,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov,
                maximize,
                is_first_step);
  }
};

template <>
struct FusedSgdEncodingFunctor<false> {
  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const double weight_decay,
                  const double lr,
                  const bool maximize) const {
    mtl_setArgs(computeEncoder, tensorArgumentBuffer, metadata_arguments, weight_decay, lr, maximize);
  }

  void operator()(id<MTLComputeCommandEncoder>& computeEncoder,
                  id<MTLBuffer>& tensorArgumentBuffer,
                  const MetadataArguments& metadata_arguments,
                  const double weight_decay,
                  const at::Tensor& lr,
                  const bool maximize) const {
    mtl_setArgs(computeEncoder, tensorArgumentBuffer, metadata_arguments, weight_decay, lr, maximize);
  }
};

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getFusedAdamCPLState(const std::string& fname);
template <int depth, uint32_t kThreadGroupSize, typename encoder_func_t, typename... ArgTypes>
static void multi_tensor_apply_for_fused_optimizer(const std::string& kernel_name,
                                                   std::vector<std::vector<at::Tensor>>& tensor_lists,
                                                   at::TensorList state_steps,
                                                   encoder_func_t encode,
                                                   ArgTypes... args) {
  const auto num_tensors = tensor_lists[0].size();

  if (num_tensors == 0) {
    return;
  }

  TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists has to match the depth");
  for (const auto& d : c10::irange(depth)) {
    const auto scalar_type = tensor_lists[d][0].scalar_type();
    TORCH_CHECK(scalar_type == kFloat || scalar_type == kHalf || scalar_type == kBFloat16,
                "Only float, bfloat and half are supported");
  }

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();

  // Remove comment for debugging
  /*
  mpsStream->addCompletedHandler(^(id<MTLCommandBuffer> cb) {
    [cb.logs enumerateObjectsUsingBlock:^(NSString* log, NSUInteger idx, BOOL* stop) {
      NSLog(@"MPSStream: %@", log);
      }
    ];
  });
  */

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto [fusedOptimizerPSO, fusedOptimizerFunc] = getFusedAdamCPLState(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(fusedOptimizerPSO, kernel_name, {tensor_lists[0]});

      [computeEncoder setComputePipelineState:fusedOptimizerPSO];

      // BufferIndex is the index in the kernel function
      auto tensorArgumentEncoder = [[fusedOptimizerFunc newArgumentEncoderWithBufferIndex:0] autorelease];
      id<MTLBuffer> tensorArgumentBuffer = [[device newBufferWithLength:tensorArgumentEncoder.encodedLength
                                                                options:0] autorelease];
      [tensorArgumentEncoder setArgumentBuffer:tensorArgumentBuffer offset:0];

      int64_t tensor_loc = 0;
      int64_t threadgroup_loc = 0;
      MetadataArguments metadata_arguments;

      for (const auto tensor_index : c10::irange(num_tensors)) {
        // short-circuit to avoid adding empty tensors to tensorListMeta
        if (tensor_lists[0][tensor_index].numel() == 0) {
          continue;
        }

        for (const auto& d : c10::irange(depth)) {
          mtl_setBuffer(tensorArgumentEncoder, tensor_lists[d][tensor_index], d * kmaxTensors + tensor_loc);
          [computeEncoder useResource:getMTLBufferStorage(tensor_lists[d][tensor_index])
                                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        if (!state_steps.empty()) {
          mtl_setBuffer(tensorArgumentEncoder, state_steps[tensor_index], depth * kmaxTensors + tensor_loc);
          [computeEncoder useResource:getMTLBufferStorage(state_steps[tensor_index]) usage:MTLResourceUsageRead];
        }
        metadata_arguments.numels[tensor_loc] = tensor_lists[0][tensor_index].numel();

        tensor_loc++;

        const auto numel = tensor_lists[0][tensor_index].numel();
        const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
        TORCH_CHECK(chunks > -1);

        for (const auto& chunk : c10::irange(chunks)) {
          metadata_arguments.threadgroup_to_tensor[threadgroup_loc] = tensor_loc - 1;
          metadata_arguments.threadgroup_to_chunk[threadgroup_loc] = chunk;

          threadgroup_loc++;

          const auto tensor_full = tensor_loc == kmaxTensors && chunk == chunks - 1;
          // Reach the maximum threadgroups per dispatch
          const auto blocks_full = threadgroup_loc == kmaxThreadGroups;

          if (tensor_full || blocks_full) {
            encode(computeEncoder, tensorArgumentBuffer, metadata_arguments, args...);
            MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
            uint32_t maxThreadsPerGroup = [fusedOptimizerPSO maxTotalThreadsPerThreadgroup];
            MTLSize threadGroupSize = MTLSizeMake(std::min(maxThreadsPerGroup, kThreadGroupSize), 1, 1);
            [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

            // Reset
            threadgroup_loc = 0;
            if (chunk == chunks - 1) {
              // last chunk
              tensor_loc = 0;
              tensorArgumentBuffer = [[device newBufferWithLength:tensorArgumentEncoder.encodedLength
                                                          options:0] autorelease];
              [tensorArgumentEncoder setArgumentBuffer:tensorArgumentBuffer offset:0];
            } else {
              // reuse the current tensor since the current one isn't done.
              metadata_arguments.numels[0] = metadata_arguments.numels[tensor_loc - 1];

              tensorArgumentBuffer = [[device newBufferWithLength:tensorArgumentEncoder.encodedLength
                                                          options:0] autorelease];
              [tensorArgumentEncoder setArgumentBuffer:tensorArgumentBuffer offset:0];

              for (const auto& d : c10::irange(depth)) {
                mtl_setBuffer(tensorArgumentEncoder, tensor_lists[d][tensor_index], d * kmaxTensors);
                [computeEncoder useResource:getMTLBufferStorage(tensor_lists[d][tensor_index])
                                      usage:MTLResourceUsageWrite | MTLResourceUsageRead];
              }
              if (!state_steps.empty()) {
                mtl_setBuffer(tensorArgumentEncoder, state_steps[tensor_index], depth * kmaxTensors);
                [computeEncoder useResource:getMTLBufferStorage(state_steps[tensor_index]) usage:MTLResourceUsageRead];
              }
              tensor_loc = 1;
            }
          }
        }
      }

      if (threadgroup_loc != 0) {
        encode(computeEncoder, tensorArgumentBuffer, metadata_arguments, args...);
        MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
        uint32_t maxThreadsPerGroup = [fusedOptimizerPSO maxTotalThreadsPerThreadgroup];
        MTLSize threadGroupSize = MTLSizeMake(std::min(maxThreadsPerGroup, kThreadGroupSize), 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
      }

      getMPSProfiler().endProfileKernel(fusedOptimizerPSO);
    }
  });
}

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getAmpCPLState(const std::string& fname);
template <int depth, typename... ArgTypes>
void multi_tensor_apply(const std::string& kernel_name,
                        std::vector<std::vector<at::Tensor>>& tensor_lists,
                        ArgTypes... args) {
  const auto num_tensors = tensor_lists[0].size();
  if (num_tensors == 0) {
    return;
  }

  TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists must match depth.");

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto [pipeline, function] = getAmpCPLState(kernel_name);
      [computeEncoder setComputePipelineState:pipeline];

      id<MTLArgumentEncoder> argumentEncoder = [function newArgumentEncoderWithBufferIndex:0];
      auto tensorArgumentBuffer = [[device newBufferWithLength:argumentEncoder.encodedLength options:0] autorelease];
      [argumentEncoder setArgumentBuffer:tensorArgumentBuffer offset:0];

      int tensor_loc = 0;
      int threadgroup_loc = 0;
      MetadataArguments metadata_arguments;
      std::memset(&metadata_arguments, 0, sizeof(metadata_arguments));

      for (size_t t = 0; t < num_tensors; t++) {
        if (tensor_lists[0][t].numel() == 0)
          continue;

        // bind each tensor in this list to the correct slots across depths
        for (int d = 0; d < depth; d++) {
          mtl_setBuffer(argumentEncoder, tensor_lists[d][t], d * kmaxTensors + tensor_loc);
          [computeEncoder useResource:getMTLBufferStorage(tensor_lists[d][t])
                                usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }

        // save number of elements for this tensor
        metadata_arguments.numels[tensor_loc] = tensor_lists[0][t].numel();
        int currentTensorIndex = tensor_loc;
        tensor_loc++;

        const auto numel = tensor_lists[0][t].numel();
        const auto chunks = numel / kChunkSize + ((numel % kChunkSize) ? 1 : 0);

        // process tensor in chunks based on max chunk size
        for (uint chunk = 0; chunk < chunks; chunk++) {
          metadata_arguments.threadgroup_to_tensor[threadgroup_loc] = currentTensorIndex;
          metadata_arguments.threadgroup_to_chunk[threadgroup_loc] = chunk;
          threadgroup_loc++;

          // dispatch when we've filled the threadgroup array or finished the chunks
          const bool dispatch_now = (threadgroup_loc == kmaxThreadGroups) || (chunk == chunks - 1);
          if (dispatch_now) {
            // check for a partial dispatch (i.e. more chunks remain for the current tensor)
            bool partial = (chunk != chunks - 1);
            uint carried_numels = 0;
            if (partial) {
              carried_numels = metadata_arguments.numels[currentTensorIndex];
            }

            mtl_setArgs(computeEncoder, tensorArgumentBuffer, metadata_arguments, args...);
            MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
            uint32_t maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
            MTLSize threadGroupSize = MTLSizeMake(std::min(maxThreads, (uint32_t)64), 1, 1);
            [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

            // prepare for the next batch: reset threadgroup count and create a new buffer
            threadgroup_loc = 0;
            tensorArgumentBuffer = [[device newBufferWithLength:argumentEncoder.encodedLength options:0] autorelease];
            [argumentEncoder setArgumentBuffer:tensorArgumentBuffer offset:0];

            if (partial) {
              // for a partial dispatch, rebind the partially processed tensor to slot 0
              // so that its metadata is in the correct location
              for (int d = 0; d < depth; d++) {
                mtl_setBuffer(argumentEncoder, tensor_lists[d][t], d * kmaxTensors + 0);
                [computeEncoder useResource:getMTLBufferStorage(tensor_lists[d][t])
                                      usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
              }
              metadata_arguments.numels[0] = carried_numels;
              // the currently processed tensor now lives at index 0
              currentTensorIndex = 0;
              tensor_loc = 1;
            } else {
              tensor_loc = 0;
            }
          }
        }
      }

      if (threadgroup_loc != 0) {
        mtl_setArgs(computeEncoder, tensorArgumentBuffer, metadata_arguments, args...);
        MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
        uint32_t maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
        MTLSize threadGroupSize = MTLSizeMake(std::min(maxThreads, static_cast<uint32_t>(64)), 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
      }
    }
  });
}

} // namespace at::native::mps
