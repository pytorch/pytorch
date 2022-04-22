#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mma_type.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

namespace {

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

void checkIntValue(
    ExpressionEvaluator& evaluator,
    Val* val,
    Int::ScalarType expected_value) {
  TORCH_CHECK(val->isAnInt());
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

void checkIntValue(
    kir::ExpressionEvaluator& evaluator,
    const Val* val,
    Int::ScalarType expected_value) {
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

bool cudaArchGuardShouldSkip(int required_major, int required_minor) {
  int capability_major = at::cuda::getCurrentDeviceProperties()->major;
  int capability_minor = at::cuda::getCurrentDeviceProperties()->minor;

  if (capability_major < required_major ||
      (capability_major == required_major &&
       capability_minor < required_minor)) {
    return true;
  }
  return false;
}

#define NVFUSER_TEST_CUDA_ARCH_GUARD(REQUIRED_MAJOR, REQUIRED_MINOR)          \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR)) {              \
    GTEST_SKIP() << "Requires GPU capability above " << REQUIRED_MAJOR << "." \
                 << REQUIRED_MINOR << " to run.\n";                           \
  }

} // namespace

// MMA unit test for a single instruction tile. VoltaTT
TEST_F(NVFuserTest, FusionVoltaMMATT_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M,K]
  auto tv0 = makeConcreteTensor({16, 4}, DataType::Half);
  // [K,N]
  auto tv1 = makeConcreteTensor({4, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,K,N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

  fusion.addOutput(tv2);

  // TODO: should be able to completely remove it
  //  in a follow up.
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 4);
  gemm_tile.warp_tile = GemmTile(16, 16, 4);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);
  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::TT);
  tv2->configureMma(mma_builder.build());

  // Write A to smem
  auto tv0cw = tv0b->cache_after();
  // Read A from smem
  auto tv0cr = tv0cw->cache_after();

  // Write B to smem
  auto tv1cw = tv1b->cache_after();

  // Read B from smem
  auto tv1cr = tv1cw->cache_after();

  // Register accumulator
  auto tv2c = tv2->cache_before();

  // [M,K,N]->[M,N,K]
  tv0cr->reorder({{-2, -1}, {-1, -2}});

  // Schedule the instruction tile loops, which is the only
  //  part we have in this unit test.
  // Assumes last 3 dims are mnk
  // The innermost loops are dictated by the type of mma used,
  //   the scheduler needs to use mma_util::WarpMmaSwizzler to
  //   get the right thread swizzle. Currently this is the only
  //   method allowed to schedule the 3/2 inner most loops of
  //   mma input/output.
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());

  // [M,K,N]->[M,N,K]
  tv1cr->reorder({{-2, -1}, {-1, -2}});
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // [M,K,N]->[M,N,K]
  tv2c->reorder({{-2, -1}, {-1, -2}});

  // Schedule the output instruction tile.
  // Assumes last 3 dims are mnk
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  // Set memory type.
  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 4}, options);
  auto t1 = at::randn({4, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test for a single instruction tile. VoltaTN
TEST_F(NVFuserTest, FusionVoltaMMATN_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M,K]
  auto tv0 = makeConcreteTensor({16, 4}, DataType::Half);
  // [N,K]
  auto tv1 = makeConcreteTensor({16, 4}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,N,K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  // TODO: should be able to completely remove it
  //  in a follow up.
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 4);
  gemm_tile.warp_tile = GemmTile(16, 16, 4);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);

  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::TN);

  tv2->configureMma(mma_builder.build());

  auto tv0cw = tv0b->cache_after();
  auto tv0cr = tv0cw->cache_after();
  auto tv1cw = tv1b->cache_after();
  auto tv1cr = tv1cw->cache_after();
  auto tv2c = tv2->cache_before();

  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 4}, options);
  auto t1 = at::randn({16, 4}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));
  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test for a single instruction tile. VoltaNT
TEST_F(NVFuserTest, FusionVoltaMMANT_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K,M]
  auto tv0 = makeConcreteTensor({4, 16}, DataType::Half);
  // [K,N]
  auto tv1 = makeConcreteTensor({4, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K,M,N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {0});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 4);
  gemm_tile.warp_tile = GemmTile(16, 16, 4);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);

  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::NT);

  tv2->configureMma(mma_builder.build());

  auto tv0cw = tv0b->cache_after();
  auto tv0cr = tv0cw->cache_after();
  auto tv1cw = tv1b->cache_after();
  auto tv1cr = tv1cw->cache_after();
  auto tv2c = tv2->cache_before();

  // To MNK
  tv0cr->reorder({{0, 2}, {1, 0}, {2, 1}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());

  // To MNK
  tv1cr->reorder({{0, 2}, {1, 0}, {2, 1}});
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  tv2c->reorder({{0, 2}, {1, 0}, {2, 1}});
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 16}, options);
  auto t1 = at::randn({4, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = t0.t().to(at::kFloat).matmul(t1.to(at::kFloat));
  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// Gemm test for Volta MMA: TT
//  This is the only example that is fully manual,
//    the rest of them are facilitated by gemm utils.
TEST_F(NVFuserTest, FusionVoltaMatMulTT_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Keep multiples of 8 to keep vectorizable.
  int M = 264, N = 120, K = 248;

  // [M,K]
  auto tv0 = makeContigTensor(2, DataType::Half);
  // [K,N]
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,K,N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {true, false, false});

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);

  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::TT);

  tv2->configureMma(mma_builder.build());

  auto tv0r = tv0->cache_after();
  auto tv1r = tv1->cache_after();
  auto tv0cw = tv0b->cache_after();
  auto tv0cr = tv0cw->cache_after();
  auto tv1cw = tv1b->cache_after();
  auto tv1cr = tv1cw->cache_after();
  auto tv2c = tv2->cache_before();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv2c, 3);
  tv1r->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------

  //       -3   -2  -1
  //[...    M,   N,  K]
  // Distribute warp tile: accumulator reg
  tv2c->split(-3, gemm_tile.warp_tile.m);
  tv2c->split(-2, gemm_tile.warp_tile.n);

  //  -5   -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw   K]
  tv2c->split(-4, gemm_tile.instruction_tile.m);
  tv2c->split(-2, gemm_tile.instruction_tile.n);
  tv2c->split(-1, gemm_tile.instruction_tile.k);

  //   -8  -7 -6 -5 -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni Ko Ki]
  tv2c->reorder({{-7, -5}, {-6, -3}, {-5, -7}, {-3, -2}, {-2, -6}});
  //   -8  -7  -6 -5 -4 -3 -2 -1
  // [Mwo  Nwo Ko Mw Nw Mi Ni Ki]

  // Distribute warp tile: output tensor
  tv2->split(-2, gemm_tile.warp_tile.m);
  tv2->split(-1, gemm_tile.warp_tile.n);

  //  -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw ]
  tv2->split(-3, gemm_tile.instruction_tile.m);
  tv2->split(-1, gemm_tile.instruction_tile.n);

  //  -6 -5  -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni]
  tv2->reorder({{-5, -4}, {-4, -2}, {-3, -5}, {-2, -3}});
  //  -6 -5  -4 -3 -2 -1
  // [Mwo Nwo Mw Nw Mi Ni]

  //           -8   -7  -6 -5 -4 -3 -2 -1
  // [Mo No Ko Mwo  Nwo Kwo Mw Nw Mi Ni Ki]

  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,No,Ko,M,N,K]
  tv0cw->reorder({
      {-3, -2},
      {-2, -3},
  });
  // [Mo,No,Ko,N,M,K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;
  int num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;
  int vector_word = 8;

  // Smem write
  tv0cw->split(-1, num_of_thread * vector_word);
  tv0cw->split(-1, 8);
  // [..., thread, vec]
  // distribute to warp:
  tv0cw->split(-2, 32);
  tv0cw->split(-3, warp_dims.n * warp_dims.k);

  tv0cw->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0cw->axis(-2)->parallelize(ParallelType::TIDx);
  tv0cw->axis(-3)->parallelize(ParallelType::TIDy);
  tv0cw->axis(-4)->parallelize(ParallelType::TIDz);

  // Gmem read (reg staging)
  tv0r->split(-1, num_of_thread * vector_word);
  tv0r->split(-1, 8);
  // [..., thread, vec]
  // distribute to warp:
  tv0r->split(-2, 32);
  tv0r->split(-3, warp_dims.n * warp_dims.k);

  tv0r->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0r->axis(-2)->parallelize(ParallelType::TIDx);
  tv0r->axis(-3)->parallelize(ParallelType::TIDy);
  tv0r->axis(-4)->parallelize(ParallelType::TIDz);

  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [Mo,No,Ko,M,N,K]
  tv1r->reorder({
      {-1, -2},
      {-2, -1},
  });
  tv1cw->reorder({
      {-1, -2},
      {-2, -1},
  });
  // [Mo,No,Ko,M,K,N]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [Mo,No,Ko,i,wy,wx,v]
  tv1r->split(-1, num_of_thread * vector_word);
  tv1r->split(-1, 8);
  // [..., thread, vec]
  // distribute to warp:
  tv1r->split(-2, 32);
  tv1r->split(-3, warp_dims.n * warp_dims.k);

  tv1r->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1r->axis(-2)->parallelize(ParallelType::TIDx);
  tv1r->axis(-3)->parallelize(ParallelType::TIDy);
  tv1r->axis(-4)->parallelize(ParallelType::TIDz);

  tv1cw->split(-1, num_of_thread * vector_word);
  tv1cw->split(-1, 8);
  // [..., thread, vec]
  // distribute to warp:
  tv1cw->split(-2, 32);
  tv1cw->split(-3, warp_dims.n * warp_dims.k);

  tv1cw->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1cw->axis(-2)->parallelize(ParallelType::TIDx);
  tv1cw->axis(-3)->parallelize(ParallelType::TIDy);
  tv1cw->axis(-4)->parallelize(ParallelType::TIDz);

  tv1cw->setMemoryType(MemoryType::Shared);

  // Schedule mma input
  // ---------------------------------------------------------------------------

  // Use WarpMmaSwizzler for the innermost instruction tile.(Mi, Ni, Ki)
  //           -8   -7  -6 -5 -4 -3 -2 -1
  // [Mo No Ko Mwo  Nwo Kwo Mw Nw Mi Ni Ki]
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // Schedule mma output
  // ---------------------------------------------------------------------------
  // Use WarpMmaSwizzler for the innermost instruction tile (Mi,Ni, Ki) on
  // output
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  //  -6 -5  -4 -3 -2 -1
  // [Mwo Nwo Mw Nw Mi Ni]
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  // Inline broadcast with smem write.
  tv0b->computeAt(tv0cw, -2);
  tv1b->computeAt(tv1cw, -2);

  // Vectorize smem read
  tv0cr->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);

  // Parallelize
  //  0   1  2  3    4   5  6  7  8  9  10
  // [Mo No Ko Mwo  Nwo Kw Mw Nw (Mi Ni Ki)]
  tv2c->axis(3)->parallelize(ParallelType::TIDz);
  tv2c->axis(4)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({K, N}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  TORCH_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Gemm test for Volta MMA: TN
TEST_F(NVFuserTest, FusionVoltaMatMulTN_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);
  int M = 120, N = 264, K = 56;

  // [M,K]
  auto tv0 = makeContigTensor(2, DataType::Half);
  // [N,K]
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,N,K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);

  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::TN);

  tv2->configureMma(mma_builder.build());

  auto tv0r = tv0->cache_after();
  auto tv1r = tv1->cache_after();
  auto tv0cw = tv0b->cache_after();
  auto tv0cr = tv0cw->cache_after();
  auto tv1cw = tv1b->cache_after();
  auto tv1cr = tv1cw->cache_after();
  auto tv2c = tv2->cache_before();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv2c, 3);
  tv1r->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  scheduler_utils::matmul_utils::scheduleWarpTileWithNoReduction(
      tv2, gemm_tile);
  //           -8   -7  -6 -5 -4 -3 -2 -1
  // [Mo No Ko Mwo  Nwo Kwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,No,Ko,M,N,K]
  tv0cw->reorder({
      {-3, -2},
      {-2, -3},
  });
  // [Mo,No,Ko,N,M,K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv0cw, gemm_tile, 8);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv0r, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [Mo,No,Ko,M,N,K]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [Mo,No,Ko,i,wy,wx,v]
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv1cw, gemm_tile, 8);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv1r, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // Schedule mma output
  // ---------------------------------------------------------------------------
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  tv0b->computeAt(tv0cw, -2);
  tv1b->computeAt(tv1cw, -2);

  tv0cr->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);
  // Parallelize
  //  0   1  2  3    4   5  6  7  8  9  10
  // [Mo No Ko Mwo  Nwo Kw Mw Nw (Mi Ni Ki)]
  tv2c->axis(3)->parallelize(ParallelType::TIDz);
  tv2c->axis(4)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat).t());
  TORCH_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Gemm test for Volta MMA: NT
TEST_F(NVFuserTest, FusionVoltaMatMulNT_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(7, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);
  int M = 240, N = 320, K = 136;

  // [K,M]
  auto tv0 = makeContigTensor(2, DataType::Half);
  // [K,N]
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K,M,N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {0});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 16, 4);

  auto mma_builder = MmaBuilder(MmaOptions::MacroType::Volta_16_16_4, gemm_tile)
                         .layout(MmaOptions::MmaInputLayout::NT);

  tv2->configureMma(mma_builder.build());

  auto tv0r = tv0->cache_after();
  auto tv1r = tv1->cache_after();
  auto tv0cw = tv0b->cache_after();
  auto tv0cr = tv0cw->cache_after();
  auto tv1cw = tv1b->cache_after();
  auto tv1cr = tv1cw->cache_after();
  auto tv2c = tv2->cache_before();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv2c, 3);
  tv1r->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  scheduler_utils::matmul_utils::scheduleWarpTileWithNoReduction(
      tv2, gemm_tile);
  //           -8   -7  -6 -5 -4 -3 -2 -1
  // [Mo No Ko Mwo  Nwo Kwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,No,Ko,M,N,K]
  tv0cw->reorder({{-3, -1}, {-2, -3}, {-1, -2}});
  // [Mo,No,Ko,N,K,M]
  tv0cw->merge(-2);

  // [Mo,No,M,K]
  tv0r->reorder({{-2, -1}, {-1, -2}});
  // [Mo,No,K,M]
  tv0r->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv0cw, gemm_tile, 8);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv0r, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [Mo,No,Ko,M,N,K]
  tv1cw->reorder({{-2, -1}, {-1, -2}});
  tv1r->reorder({{-2, -1}, {-1, -2}});
  // [Mo,No,Ko,M,K,N]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [Mo,No,Ko,i,wy,wx,v]
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv1cw, gemm_tile, 8);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      tv1r, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  // ---------------------------------------------------------------------------
  // [...M,N,K]
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // Schedule mma output
  // ---------------------------------------------------------------------------
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::NotOperand).build());

  tv0b->computeAt(tv0cw, -2);
  tv1b->computeAt(tv1cw, -2);

  tv0cr->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);
  // Parallelize
  //  0   1  2  3    4   5  6  7  8  9  10
  // [Mo No Ko Mwo  Nwo Kw Mw Nw (Mi Ni Ki)]
  tv2c->axis(3)->parallelize(ParallelType::TIDz);
  tv2c->axis(4)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({K, M}, options);
  auto t1 = at::randn({K, N}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = t0.to(at::kFloat).t().matmul(t1.to(at::kFloat));

  TORCH_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

} // namespace jit
} // namespace torch

#endif
