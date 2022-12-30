#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor_utils.h>
#include <fusion.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

// This is for internal testing only and is intended to be used as a template to
// compile and run an external source file. By default, it should just
// return immediately, but if PYTORCH_NVFUSER_EXTERNAL_SRC is defined,
// the file specified by the env var is loaded and compiled.
TEST_F(NVFuserTest, FusionExternalSrc_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  FusionExecutor fe;

  // By default, this env var should not be defined. To test using an
  // external source file, set it to the path to the external source
  // file.
  auto path = std::getenv("PYTORCH_NVFUSER_EXTERNAL_SRC");
  if (path == nullptr) {
    return;
  }

  std::cout << "Compiling " << path << std::endl;
  std::ifstream cuda_src(path);
  std::stringstream buffer;
  buffer << cuda_src.rdbuf();
  std::string cuda_src_str = buffer.str();

  fe.compileRtc(cuda_src_str, "CudaCodeGen::kernel1", true);

  // The following is a sample launch pattern of the compiled
  // kernel. It must be adapted for each particular source file.

  int N = 256, H = 7, W = 7, C = 512;
  int tidx = 16, tidy = 16, bidx = 4, bidy = 16;

  LaunchParams lp(bidx, bidy, 1, tidx, tidy, 1);
  lp.setSmem(tidx * tidy * sizeof(float) * 3);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto options_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, H, W, C};

  auto t0 = at::randn(input_shape, options_half);
  auto t7 = at::zeros(input_shape, options_half);

  auto t1 = t0.to(at::kFloat);
  auto t2 = t1.mean({0, 1, 2});
  auto t3 = t2.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto ref = t1 - t3;

  float read_write_bytes =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 2 * 2;

  for (int i = 0; i < 5; ++i) {
    auto t14 = at::zeros_like(t0, options_float);
    auto t15 = at::zeros_like(t0, options_float);
    auto t16 = at::zeros_like(t0, options_int);
    auto t17 = at::zeros({8}, options_long);
    clearL2Cache();
    std::cout << "Launching the kernel" << std::endl;
    float elapsed_time_ms =
        fe.runRtc(lp, {t0, t7, t14, t15, t16, t17}, KernelIndexMode::INT32);
    std::cout << "kernel run in " << elapsed_time_ms << " ms, achieved "
              << (read_write_bytes / elapsed_time_ms / 1000.0 / 1000.0)
              << " GB/s" << std::endl;

    auto fusion_out = t7.to(at::kFloat);
    std::cout << "Max diff: " << (ref - fusion_out).abs().max() << std::endl;
    TORCH_CHECK(ref.allclose(fusion_out, /*rtol*/ 0.005, /*atol*/ 0.5));
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
