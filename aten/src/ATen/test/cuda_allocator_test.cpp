#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

std::unordered_map<void*, ssize_t> allocation_sizes;

void* logging_malloc(ssize_t size, int device, cudaStream_t stream) {
    void* ptr;
    cudaMalloc(&ptr, size);
    std::cout << "alloc ptr=" << ptr << " size=" << size << " device=" << device
              << " stream=" << stream << std::endl;
    allocation_sizes[ptr] = size;
    return ptr;
}

void logging_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    std::cout << "free  ptr=" << ptr << " size=" << size << " device=" << device
              << " stream=" << stream << std::endl;
    // Print out any frees that don't match the allocation sizes
    if (allocation_sizes.find(ptr) != allocation_sizes.end()) {
        if (allocation_sizes[ptr] != size) {
            std::cout << "*** ERROR: free mismatch: " << ptr << " size=" << size
                      << " expected=" << allocation_sizes[ptr] << std::endl;
        }
    } else {
        std::cout << "WARNING: free of unknown ptr=" << ptr << std::endl;
    }
    cudaFree(ptr);
    allocation_sizes.erase(ptr);
}

TEST(TestTorchUnique, UniqueComparisonTest) {
    auto custom_allocator =
        torch::cuda::CUDAPluggableAllocator::createCustomAllocator(logging_malloc, logging_free);
    torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(custom_allocator);
    // Run the command 3 times; the first 2 will pass and the third invocation will have
    // different sizes in alloc and free
    for (int i = 0; i < 3; ++i) {
        LOG(INFO) << "Starting test " << i;
        // Initialize simple sorted tensor with repeats
        at::Tensor sorted_tensor =
            at::tensor({0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 5},
                          at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

        LOG(INFO) << "Starting unique_consecutive";
        // This operation will call malloc/free with different sizes on the same pointer
        auto unique_dim_result = at::unique_consecutive(sorted_tensor, false, true, 0);
        LOG(INFO) << "Finished unique_consecutive";
        // Everything below is only there to validate correct results
        auto unique_dim_values = std::get<0>(unique_dim_result);
        auto unique_dim_counts = std::get<2>(unique_dim_result);

        // Check tensor sizes
        EXPECT_EQ(unique_dim_values.size(0), 5);
        EXPECT_EQ(unique_dim_counts.size(0), 5);

        // Copy to CPU before accessing elements
        at::Tensor cpu_values = unique_dim_values.cpu();
        at::Tensor cpu_counts = unique_dim_counts.cpu();

        // Use accessors on the CPU tensors
        auto values_accessor = cpu_values.accessor<float, 1>();
        auto counts_accessor = cpu_counts.accessor<int64_t, 1>();

        // Check individual values using accessors
        EXPECT_EQ(values_accessor[0], 0.0f);
        EXPECT_EQ(values_accessor[1], 1.0f);
        EXPECT_EQ(values_accessor[2], 2.0f);
        EXPECT_EQ(values_accessor[3], 3.0f);
        EXPECT_EQ(values_accessor[4], 5.0f);

        // Check count values using accessors
        EXPECT_EQ(counts_accessor[0], 3);
        EXPECT_EQ(counts_accessor[1], 2);
        EXPECT_EQ(counts_accessor[2], 1);
        EXPECT_EQ(counts_accessor[3], 4);
        EXPECT_EQ(counts_accessor[4], 1);
    }
}

TEST(AllocatorTestCUDA, test_clone) {
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}
