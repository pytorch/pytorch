#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// this sample custom kernel is taken from:
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
static const char* CUSTOM_KERNEL = R"MPS_ADD_ARRAYS(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
)MPS_ADD_ARRAYS";

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static inline id<MTLBuffer> getMTLBuffer(const c10::DataPtr& data_ptr) {
  return __builtin_bit_cast(id<MTLBuffer>, data_ptr.get());
}

TEST(MPSObjCInterfaceTest, MPSPlacementHeapCoalescing) {
  ASSERT_TRUE(torch::mps::is_available());

  auto* allocator = at::mps::getIMPSAllocator();
  allocator->emptyCache();
  constexpr size_t block_size = 8 * 1024 * 1024;

  auto first = allocator->allocate(block_size);
  auto second = allocator->allocate(block_size);
  auto third = allocator->allocate(block_size);
  id<MTLHeap> heap = [getMTLBuffer(first) heap];
  ASSERT_NE(heap, nil);
  ASSERT_EQ([heap type], MTLHeapTypePlacement);
  ASSERT_EQ([getMTLBuffer(second) heap], heap);
  ASSERT_EQ([getMTLBuffer(third) heap], heap);
  const NSUInteger second_offset = [getMTLBuffer(second) heapOffset];
  ASSERT_EQ(second_offset, [getMTLBuffer(first) heapOffset] + [getMTLBuffer(first) length]);
  ASSERT_EQ([getMTLBuffer(third) heapOffset], second_offset + [getMTLBuffer(second) length]);

  second.clear();
  third.clear();
  auto merged = allocator->allocate(2 * block_size);
  ASSERT_EQ([getMTLBuffer(merged) heap], heap);
  ASSERT_EQ([getMTLBuffer(merged) heapOffset], second_offset);

  first.clear();
  merged.clear();
  allocator->emptyCache();
}

TEST(MPSObjCInterfaceTest, MPSPlacementHeapSplitting) {
  ASSERT_TRUE(torch::mps::is_available());

  auto* allocator = at::mps::getIMPSAllocator();
  allocator->emptyCache();
  const size_t reserved = allocator->getTotalAllocatedMemory();

  // small allocations are placed by the allocator as well
  auto small = allocator->allocate(4096);
  ASSERT_EQ([[getMTLBuffer(small) heap] type], MTLHeapTypePlacement);

  // a free range larger than the request is cut down to it, and the remainder
  // stays available for the next allocation
  constexpr size_t block_size = 8 * 1024 * 1024;
  auto whole = allocator->allocate(block_size);
  id<MTLHeap> heap = [getMTLBuffer(whole) heap];
  const NSUInteger offset = [getMTLBuffer(whole) heapOffset];
  whole.clear();

  auto head = allocator->allocate(block_size / 2);
  auto tail = allocator->allocate(block_size / 2);
  ASSERT_EQ([getMTLBuffer(head) heap], heap);
  ASSERT_EQ([getMTLBuffer(head) heapOffset], offset);
  ASSERT_EQ([getMTLBuffer(tail) heap], heap);
  ASSERT_EQ([getMTLBuffer(tail) heapOffset], offset + block_size / 2);

  // heaps are handed back to the system once no allocation is left in them
  small.clear();
  head.clear();
  tail.clear();
  allocator->emptyCache();
  ASSERT_EQ(allocator->getTotalAllocatedMemory(), reserved);
}

TEST(MPSObjCInterfaceTest, MPSCustomKernel) {
  const unsigned int tensor_length = 100000UL;

  // fail if mps isn't available
  ASSERT_TRUE(torch::mps::is_available());

  torch::Tensor cpu_input1 = torch::randn({tensor_length}, at::device(at::kCPU));
  torch::Tensor cpu_input2 = torch::randn({tensor_length}, at::device(at::kCPU));
  torch::Tensor cpu_output = cpu_input1 + cpu_input2;

  torch::Tensor mps_input1 = cpu_input1.detach().to(at::kMPS);
  torch::Tensor mps_input2 = cpu_input2.detach().to(at::kMPS);
  torch::Tensor mps_output = torch::empty({tensor_length}, at::device(at::kMPS));

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    size_t numThreads = mps_output.numel();
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource: [NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                              options: nil
                                                                error: &error];
    TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    id<MTLFunction> customFunction = [customKernelLibrary newFunctionWithName: @"add_arrays"];
    TORCH_CHECK(customFunction, "Failed to create function state object for the kernel");

    id<MTLComputePipelineState> kernelPSO = [device newComputePipelineStateWithFunction: customFunction error: &error];
    TORCH_CHECK(kernelPSO, error.localizedDescription.UTF8String);

    // Get a reference of the MPSStream MTLCommandBuffer.
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // Get a reference of the MPSStream dispatch_queue. This is used for CPU side synchronization while encoding.
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
    dispatch_sync(serialQueue, ^(){
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState: kernelPSO];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input1) offset:0 atIndex:0];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input2) offset:0 atIndex:1];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_output) offset:0 atIndex:2];
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      // Calculate a thread group size.
      NSUInteger threadsPerGroupSize = std::min(kernelPSO.maxTotalThreadsPerThreadgroup, numThreads);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupSize, 1, 1);

      // Encode the compute command.
      [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadGroupSize];
      [computeEncoder endEncoding];

      torch::mps::commit();
    });
  }
  // synchronize the MPS stream before reading back from MPS buffer
  torch::mps::synchronize();

  ASSERT_TRUE(at::allclose(cpu_output, mps_output.to(at::kCPU)));
}
