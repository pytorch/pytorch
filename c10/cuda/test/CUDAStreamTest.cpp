#include <gtest/gtest.h>

#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

TEST(TestStream, RegisterStreamTest) {
  {
    // Invalid device should throw
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_THROW(c10::cuda::registerCustomCUDAStream(-1, stream), c10::Error);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
  }

  {
    // Invalid(destroyed) stream should throw
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
    ASSERT_THROW(c10::cuda::registerCustomCUDAStream(0, stream), c10::Error);
  }

  {
    // Custom CUDAStream can cast back to the correct cudaStream_t
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_TRUE(c10::cuda::registerCustomCUDAStream(0, stream).stream() == stream);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
  }

  {
    // Custom CUDAStream works with StreamGuard
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    c10::cuda::CUDAStream custom_stream = c10::cuda::registerCustomCUDAStream(0, stream);
    {
      c10::cuda::CUDAStreamGuard guard(custom_stream);
      ASSERT_TRUE(c10::cuda::getCurrentCUDAStream() == custom_stream);
    }
    ASSERT_TRUE(c10::cuda::getCurrentCUDAStream() != custom_stream);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
  }
}
