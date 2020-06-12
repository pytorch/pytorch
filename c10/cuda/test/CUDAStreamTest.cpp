#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

TEST(TestStream, RegisterEventTest) {
  if (!at::cuda::is_available()) return;

  {
    // Invalid device should throw
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_THROW(c10::cuda::registerCustomCUDAStream(-1, stream), c10::Error);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
  }

  {
    // Invalid stream should throw with destroyed stream
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
    ASSERT_THROW(at::cuda::registerCustomCUDAStream(0, stream), c10::Error);
  }

  {
    // Stream can correctly cast back to the old stream
    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    ASSERT_TRUE(c10::cuda::registerCustomCUDAStream(0, stream).stream() == stream);
    ASSERT_TRUE(cudaStreamDestroy(stream) == cudaSuccess);
  }

  {
    // Stream works with StreamGuard
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
