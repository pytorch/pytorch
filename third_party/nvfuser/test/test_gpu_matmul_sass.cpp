#include <gtest/gtest.h>

#include <ops/arith.h>
#include <scheduler/matmul.h>
#include <test/test_utils.h>

#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
//
// Some useful instructions for Ampere matmul:
// - LDGDEPBAR: Global Load Dependency Barrier
// - LDGSTS: Asynchronous Global to Shared Memcopy
// - LDSM: Load Matrix from Shared Memory with Element Size Expansion
// - HMMA: Matrix Multiply and Accumulate
// - BAR: Barrier Synchronization
// - DEPBAR: Dependency Barrier

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
  params.double_buffer_options.double_buffer_smem_read = true;
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

  bool found_LDGSTS = false;
  bool found_LDSM = false;
  bool found_HMMA = false;

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
                found_LDGSTS = true;
              } else if (i.opCode() == "LDSM") {
                found_LDSM = true;
              } else if (i.opCode() == "HMMA") {
                found_HMMA = true;
              }
            }
          },
          inst);
    }
    TORCH_CHECK(found_LDGSTS);
    TORCH_CHECK(found_LDSM);
    TORCH_CHECK(found_HMMA);
  }
}

// Check the modifiers of instructions. We are particularily interested in
// load/store, mma, and sync instructions. Currently, the ground truth in this
// test's asserts are based on experimental result of this test itself. In the
// future, we should use cutlass's kernel as ground truth.
TEST_F(NVFuserTest, FusionAmpereMatmulSASSModifiersCheck_CUDA) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  bool found_LDGSTS = false;
  bool found_LDSM = false;
  bool found_HMMA = false;
  bool found_LDGDEPBAR = false;
  bool found_BAR = false;
  bool found_DEPBAR = false; // kAllSupportedMatmulLayout;
  for (auto layout : {MatmulLayout::TT}) {
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
                const std::vector<std::string> expect = {"E", "BYPASS", "128"};
                TORCH_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGSTS has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_LDGSTS = true;
              } else if (i.opCode() == "LDGDEPBAR") {
                const std::vector<std::string> expect;
                TORCH_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_LDGDEPBAR = true;
              } else if (i.opCode() == "LDSM") {
                const std::vector<std::string> expect1 = {"16", "M88", "2"};
                const std::vector<std::string> expect2 = {"16", "M88", "4"};
                const std::vector<std::string> expect3 = {"16", "MT88", "2"};
                const std::vector<std::string> expect4 = {"16", "MT88", "4"};
                TORCH_CHECK(
                    i.modifiers() == expect1 || i.modifiers() == expect2 ||
                        i.modifiers() == expect3 || i.modifiers() == expect4,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_LDSM = true;
              } else if (i.opCode() == "HMMA") {
                const std::vector<std::string> expect = {"16816", "F32"};
                TORCH_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for HMMA has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_HMMA = true;
              } else if (i.opCode() == "BAR") {
                const std::vector<std::string> expect = {"SYNC"};
                TORCH_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for BAR has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_BAR = true;
              } else if (i.opCode() == "DEPBAR") {
                const std::vector<std::string> expect = {"LE"};
                TORCH_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for DEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense and update this test.");
                found_DEPBAR = true;
              }
            }
          },
          inst);
    }
    TORCH_CHECK(found_LDGSTS);
    TORCH_CHECK(found_LDSM);
    TORCH_CHECK(found_HMMA);
    TORCH_CHECK(found_LDGDEPBAR);
    TORCH_CHECK(found_BAR);
    TORCH_CHECK(found_DEPBAR);
  }
}

// Check that all LDSM instructions has the following pattern:
//   LDSM.16.M88.2 R2,   [R213] ;
//   LDSM.16.M88.2 R136, [R213+0x200] ;
//   LDSM.16.M88.2 R138, [R213+0x400] ;
//   LDSM.16.M88.2 R140, [R213+0x600] ;
//   LDSM.16.M88.2 R142, [R213+0x800] ;
//   LDSM.16.M88.2 R144, [R213+0xa00] ;
//   LDSM.16.M88.2 R146, [R213+0xc00] ;
//   LDSM.16.M88.2 R148, [R213+0xe00] ;
TEST_F(NVFuserTest, FusionAmpereMatmulSASSRegisterUsageLDSM_CUDA) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  for (auto layout : kAllSupportedMatmulLayout) {
    std::unordered_map<std::string, std::unordered_set<int>> base_offsets;

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
              if (i.opCode() != "LDSM") {
                return;
              }
              auto args = i.args();
              TORCH_INTERNAL_ASSERT(args.size() == 2);
              std::string smem_address = args[1];
              // get base shared memory address
              std::string_view view(smem_address); // example: [R0+UR0+0x200]
              view = view.substr(1, view.size() - 2); // example: R0+UR0+0x200
              std::string_view base;
              int offset;
              using namespace std::literals;
              auto last = view.find_last_of("+"sv);
              if (last == std::string::npos ||
                  view.substr(last + 1, 2) != "0x"sv) {
                // [R0] or [R0+UR0]
                base = view;
                offset = 0;
              } else {
                // [R0+0x200] or [R0+UR0+0x200]
                base = view.substr(0, last);
                std::stringstream ss(std::string(view.substr(last + 1)));
                ss >> std::hex >> offset;
              }
              base_offsets[std::string(base)].insert(offset);
            }
          },
          inst);
    }
    for (auto& [base, offsets] : base_offsets) {
      TORCH_CHECK(
          offsets.size() > 1,
          "Expect base addresses to be used multiple times, but ",
          base,
          " is only used once");
    }
  }
}

} // namespace torch::jit
