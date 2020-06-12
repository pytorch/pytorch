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
    auto cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ASSERT_TRUE(cuda_error == cudaSuccess);
    try {
      c10::cuda::registerCustomCUDAStream(-1, stream);
      FAIL();
    } catch(c10::Error) {/* ok */}

    cuda_error = cudaStreamDestroy(stream);
    ASSERT_TRUE(cuda_error == cudaSuccess);
  }

  {
    // Invalid stream should throw with destroyed stream
    cudaStream_t stream;
    auto cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ASSERT_TRUE(cuda_error == cudaSuccess);
    cuda_error = cudaStreamDestroy(stream);
    ASSERT_TRUE(cuda_error == cudaSuccess);
    try {
      c10::cuda::CUDAStream custom_stream = at::cuda::registerCustomCUDAStream(0, stream);
      FAIL();
    } catch(c10::Error) {/* ok */}
  }

  {
    // Stream can correctly cast back to the old stream
    cudaStream_t stream;
    auto cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ASSERT_TRUE(cuda_error == cudaSuccess);
    try {
      c10::cuda::CUDAStream custom_stream = c10::cuda::registerCustomCUDAStream(0, stream);
      ASSERT_TRUE(custom_stream.stream() == stream);
    } catch(c10::Error) {/* ok */}

    cuda_error = cudaStreamDestroy(stream);
    ASSERT_TRUE(cuda_error == cudaSuccess);
  }

  {
    // Stream works with StreamGuard
    cudaStream_t stream;
    auto cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ASSERT_TRUE(cuda_error == cudaSuccess);
    try {
      c10::cuda::CUDAStream custom_stream = c10::cuda::registerCustomCUDAStream(0, stream);
      {
        c10::cuda::CUDAStreamGuard guard(custom_stream);
        c10::cuda::CUDAStream stream2 = c10::cuda::getCurrentCUDAStream();
        ASSERT_TRUE(stream2 == custom_stream);
      }
      c10::cuda::CUDAStream stream2 = c10::cuda::getCurrentCUDAStream();
      ASSERT_TRUE(stream2 != custom_stream);
    } catch(c10::Error) {/* ok */}

    cuda_error = cudaStreamDestroy(stream);
    ASSERT_TRUE(cuda_error == cudaSuccess);
  }
}
