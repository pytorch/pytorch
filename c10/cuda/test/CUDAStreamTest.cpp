#include <gtest/gtest.h>

#include <c10/cuda/CUDAStreamTest.h>

TEST(CUDASteamTest, CUDASteamTest) {
    c10::cuda::c10_cuda_stream_test();
}
