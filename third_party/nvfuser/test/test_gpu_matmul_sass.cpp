#include <gtest/gtest.h>

#include <arith.h>
#include <scheduler/matmul.h>
#include <test/test_utils.h>

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
//
// Some useful instructions for Ampere matmul:
// - LDGDEPBAR: Global Load Dependency Barrier
// - LDGSTS: Asynchronous Global to Shared Memcopy
// - LDSM: Load Matrix from Shared Memory with Element Size Expansion
// - HMMA: Matrix Multiply and Accumulate

// Tests go in torch::jit
namespace torch::jit {

namespace {

sass::Container getSASSFor(
    MatmulLayout layout,
    GemmTile cta_tile,
    GemmTile warp_tile,
    GemmTile instruction_tile,
    MmaOptions::MacroType macro,
    int M,
    int N,
    int K) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = matmul(tv0, tv1, layout);

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  gemm_tile.warp_tile = warp_tile;
  gemm_tile.instruction_tile = instruction_tile;

  auto mma_builder = MmaBuilder(macro, gemm_tile).layout(layout);

  MatmulParam params(mma_builder);
  params.tile_sizes = gemm_tile;
  params.async_gmem_load_operands = true;
  params.double_buffer_options.double_buffer_smem_write = true;
  params.double_buffer_options.smem_double_buffer_stage = 4;
  scheduleMatmul(tv2, tv0, tv1, params);

  at::manual_seed(0);
  auto inputs = fp16MatmulAtInput(M, N, K, layout);

  FusionExecutor fe;
  fe.setSaveCompiledBinaryFlag(true);
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

  TORCH_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));

  return sass::parse(fe.disassembledKernelSASS());
}

} // namespace

TEST_F(NVFuserTest, FusionAmpereMatmulSASSSanityCheck_CUDA) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  bool found_cpasync = false;
  bool found_ldmatrix = false;
  bool found_hmma = false;

  for (auto layout : kAllSupportedMatmulLayout) {
    sass::Container sass;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        sass = getSASSFor(
            layout,
            GemmTile(128, 128, 32),
            GemmTile(64, 64, 32),
            GemmTile(16, 8, 16),
            MmaOptions::MacroType::Ampere_16_8_16,
            M,
            N,
            K));
    for (auto inst : sass.code) {
      std::visit(
          [&](auto&& i) {
            using T = std::decay_t<decltype(i)>;
            if constexpr (std::is_same_v<sass::Instruction, T>) {
              if (i.opCode() == "LDGSTS") {
                found_cpasync = true;
              } else if (i.opCode() == "LDSM") {
                found_ldmatrix = true;
              } else if (i.opCode() == "HMMA") {
                found_hmma = true;
              }
            }
          },
          inst);
    }
    TORCH_CHECK(found_cpasync);
    TORCH_CHECK(found_ldmatrix);
    TORCH_CHECK(found_hmma);
  }
}

} // namespace torch::jit
