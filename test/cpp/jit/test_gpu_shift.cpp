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
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include "test_gpu_validator.h"

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
    const kir::Val* val,
    kir::Int::ScalarType expected_value) {
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

// ATen version of tensor shifting
auto shift(at::Tensor tensor, const std::vector<int>& offsets) {
  TORCH_INTERNAL_ASSERT(tensor.ndimension() == offsets.size());
  at::Tensor t = tensor;
  for (size_t i = 0; i < offsets.size(); ++i) {
    const auto offset = offsets[i];
    if (offset == 0) {
      continue;
    }
    t = t.roll(offsets[i], i);
    std::vector<at::indexing::TensorIndex> indices(
        tensor.ndimension(), Slice(0, None));
    if (offset > 0) {
      indices[i] = Slice(0, offset);
    } else {
      indices[i] = Slice(offset, None);
    }
    t.index(indices) = 0;
  }
  return t;
}

} // namespace

// Shift an input tensor
TEST(NVFuserTest, FusionShift1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = shift(tv0, {-1, 0});
  fusion.addOutput(tv1);

  auto tv2 = shift(tv0, {0, 1});
  fusion.addOutput(tv2);

  auto tv3 = shift(tv0, {2, 2});
  fusion.addOutput(tv3);

  auto tv4 = shift(tv0, {-2, -2});
  fusion.addOutput(tv4);

  int numel_x = 9;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = shift(t0, {-1, 0});
  TORCH_CHECK(t1.equal(outputs[0]));

  auto t2 = shift(t0, {0, 1});
  TORCH_CHECK(t2.equal(outputs[1]));

  auto t3 = shift(t0, {2, 2});
  TORCH_CHECK(t3.equal(outputs[2]));

  auto t4 = shift(t0, {-2, -2});
  TORCH_CHECK(t4.equal(outputs[3]));
}

// Shifts an intermediate tensor
TEST(NVFuserTest, FusionShift2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {-1, 0});
  fusion.addOutput(tv2);

  // make it a little more complex
  auto tv3 = add(tv0, new Double(3));
  auto tv4 = add(tv3, new Double(4));
  auto tv5 = shift(tv4, {-1, 0});
  auto tv6 = shift(tv4, {0, -1});
  auto tv7 = shift(tv4, {1, 0});
  auto tv8 = shift(tv4, {0, 0});
  auto tv9 = add(tv5, tv6);
  auto tv10 = add(tv9, tv7);
  auto tv11 = add(tv10, tv8);
  fusion.addOutput(tv11);

  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11}) {
    tv->setMemoryType(MemoryType::Global);
  }

  // t1 allocation: (t1.size[0] + 1) * (t1.size[1])
  // t3 allocation: (t3.size[0] + 2) * (t3.size[1] + 1)
  // t4 allocation: (t3.size[0] + 2) * (t3.size[1] + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 3 || tensor_name == 4) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          if (tensor_name == 1 && i == 1) {
            TORCH_CHECK(alloc->shape().at(i)->isA<kir::NamedScalar>());
            continue;
          }
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          TORCH_CHECK(def != nullptr && def->operation() == BinaryOpType::Add);
          TORCH_CHECK(def->as<kir::BinaryOp>()->lhs()->isA<kir::NamedScalar>());
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          if (tensor_name == 1) {
            TORCH_CHECK(i == 0);
            TORCH_CHECK(rhs_value == 1);
          } else {
            if (i == 0) {
              TORCH_CHECK(rhs_value == 2);
            } else {
              TORCH_CHECK(rhs_value == 1);
            }
          }
        }
      }
    }
  }

  int numel_x = 9;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {-1, 0});

  auto t3 = t0 + 3;
  auto t4 = t3 + 4;
  auto t5 = shift(t4, {-1, 0});
  auto t6 = shift(t4, {0, -1});
  auto t7 = shift(t4, {1, 0});
  auto t8 = shift(t4, {0, 0});
  auto t9 = t5 + t6;
  auto t10 = t9 + t7;
  auto t11 = t10 + t8;

  testValidate(&fusion, outputs, inputs, {t2, t11}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftRightOfCA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {0, 1});
  fusion.addOutput(tv2);

  tv0->computeAt(tv2, -2);

  tv1->setMemoryType(MemoryType::Global);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 100;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {0, 1});

  TORCH_CHECK(t2.allclose(outputs[0]));
}

TEST(NVFuserTest, FusionShiftLeftOfCA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(1));
  auto tv3 = shift(tv2, {-1, 0});
  auto tv4 = add(tv3, new Double(1));
  fusion.addOutput(tv4);

  tv0->computeAt(tv4, -1);

  // Lowering should trigger an assertion failure as a shifted axis is
  // found inside an allocation position.
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST(NVFuserTest, FusionShiftSplit1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {0, 1});
  auto tv3 = shift(tv1, {0, -2});
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  int split_factor = 4;
  tv2->split(-1, split_factor);
  tv3->split(-1, split_factor);

  tv0->computeAt(tv2, -2);
  tv0->computeAt(tv3, -2);

  // t1 allocation: (4 + 3)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto def =
            dynamic_cast<kir::BinaryOp*>(alloc->shape().at(0)->definition());
        auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
        TORCH_CHECK(lhs != nullptr && lhs->isConst());
        int lhs_value = *lhs->value();
        auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
        TORCH_CHECK(rhs != nullptr && rhs->isConst());
        int rhs_value = *rhs->value();
        TORCH_CHECK(lhs_value == split_factor && rhs_value == 3);
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 9;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {0, 1});
  auto t3 = shift(t1, {0, -2});

  testValidate(&fusion, outputs, inputs, {t2, t3}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftSplit2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(1));
  auto tv3 = shift(tv2, {0, -1});
  auto tv4 = shift(tv2, {0, 1});
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  auto tv6 = add(tv0, new Double(1));
  auto tv7 = shift(tv6, {0, 0});
  auto tv8 = add(tv7, new Double(1));
  fusion.addOutput(tv8);

  int split_factor = 4;

  tv5->split(-1, split_factor);
  tv8->split(-1, split_factor);

  tv0->computeAt(tv5, -2);
  tv0->computeAt(tv8, -2);

  // t1 and t2 allocation: (4 + 2)
  // t4 allocation: (4)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto def =
            dynamic_cast<kir::BinaryOp*>(alloc->shape().at(0)->definition());
        auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
        TORCH_CHECK(lhs != nullptr && lhs->isConst());
        int lhs_value = *lhs->value();
        auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
        TORCH_CHECK(rhs != nullptr && rhs->isConst());
        int rhs_value = *rhs->value();
        TORCH_CHECK(lhs_value == split_factor && rhs_value == 2);
      } else if (tensor_name == 4) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto size = dynamic_cast<kir::Int*>(alloc->shape().at(0));
        TORCH_CHECK(size != nullptr && size->isConst());
        int size_value = *size->value();
        TORCH_CHECK(size_value == split_factor);
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 9;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 2;
  auto t3 = shift(t1, {0, -1});
  auto t4 = shift(t1, {0, 1});
  auto t5 = t3 + t4;

  auto t6 = t0 + 1;
  auto t7 = t6;
  auto t8 = t7 + 1;

  testValidate(&fusion, outputs, inputs, {t5, t8}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftDoubleSplit_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(2));
  auto tv3 = shift(tv2, {0, 1});
  fusion.addOutput(tv3);

  int split_factor1 = 8;
  int split_factor2 = 4;

  tv3->split(-1, split_factor1);

  tv0->computeAt(tv3, -2);

  tv1->split(-1, split_factor2);

  // t1: [i1, i2/8, 8/4, 4]
  // t2: [i1, i2/8, 8]
  // t3: [i1, i2/8, 8]

  // t1 and t2 allocation: (split_factor1 + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto def =
            dynamic_cast<kir::BinaryOp*>(alloc->shape().at(0)->definition());
        auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
        TORCH_CHECK(lhs != nullptr && lhs->isConst());
        int lhs_value = *lhs->value();
        auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
        TORCH_CHECK(rhs != nullptr && rhs->isConst());
        int rhs_value = *rhs->value();
        TORCH_CHECK(lhs_value == split_factor1 && rhs_value == 1);
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 3;
  auto ref = shift(t1, {0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift3ptStencil_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 3-pt stencil
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  std::vector<std::vector<int>> offsets = {{-1}, {1}};

  std::vector<TensorView*> tvs;
  for (const auto& offset : offsets) {
    tvs.push_back(shift(tv0, offset));
  }

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  int split_factor = 4;

  tv_out->split(0, split_factor);

  // This seems fine but not verified yet
  // tv_out->axis(-1)->parallelize(ParallelType::Unswitch);

  auto cache = tv0->cache_after();

  tv0->computeAt(tv_out, 1);

  // Inline completely except for the cache
  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  // cache allocation: (split_factor + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == cache->name()) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto def =
            dynamic_cast<kir::BinaryOp*>(alloc->shape().at(0)->definition());
        auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
        TORCH_CHECK(lhs != nullptr && lhs->isConst());
        int lhs_value = *lhs->value();
        auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
        TORCH_CHECK(rhs != nullptr && rhs->isConst());
        int rhs_value = *rhs->value();
        TORCH_CHECK(lhs_value == split_factor && rhs_value == 2);
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = (t0 + shift(t0, {-1}) + shift(t0, {1})) / 3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift5ptStencil_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 5-pt stencil
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  std::vector<std::vector<int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  std::vector<TensorView*> tvs;
  for (const auto& offset : offsets) {
    tvs.push_back(shift(tv0, offset));
  }

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  std::vector<int> split_factor({4, 8});

  tv_out->split(-1, split_factor[1]);
  tv_out->split(0, split_factor[0]);
  tv_out->reorder({{1, 2}, {2, 1}});

  auto cache = tv0->cache_after();

  tv0->computeAt(tv_out, 2);

  // Inline completely except for the cache
  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  // cache allocation: (split_factor + 2) * (split_factor + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == cache->name()) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor[i] && rhs_value == 2);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = t0;
  for (const auto& offset : offsets) {
    ref = ref + shift(t0, offset);
  }
  ref = ref / int(offsets.size() + 1);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift9ptStencil_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 9-pt stencil
  std::vector<std::vector<int>> offsets;
  for (int i = -1; i < 2; ++i) {
    for (int j = -1; j < 2; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      offsets.push_back({i, j});
    }
  }

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  std::vector<TensorView*> tvs;
  for (const auto& offset : offsets) {
    tvs.push_back(shift(tv0, offset));
  }

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  std::vector<int> split_factor({4, 8});
  tv_out->split(-1, split_factor[1]);
  tv_out->split(0, split_factor[0]);
  tv_out->reorder({{1, 2}, {2, 1}});

  auto cache = tv0->cache_after();

  tv0->computeAt(tv_out, 2);

  // Inline completely except for the cache
  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  // This seems fine but not yet verified
  // tv_out->axis(-1)->parallelize(ParallelType::Unswitch);

  // cache allocation: (split_factor + 2) * (split_factor + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == cache->name()) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor[i] && rhs_value == 2);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = t0;
  for (const auto& offset : offsets) {
    ref = ref + shift(t0, offset);
  }
  ref = ref / int(offsets.size() + 1);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftSmemBlocking_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {0, 1});
  fusion.addOutput(tv2);

  int smem_block_factor = 32;

  tv2->split(-1, smem_block_factor);

  tv0->computeAt(tv2, -2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  // tv1 allocation: (split_factor + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == tv1->name()) {
        TORCH_CHECK(alloc->shape().size() == 1);
        for (int i = 0; i < 1; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == smem_block_factor && rhs_value == 1);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 100;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {0, 1});
  auto ref = t2;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift3ptStencilParallel_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 3-pt stencil
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  std::vector<TensorView*> tvs;
  tvs.push_back(shift(tv0, {-1}));
  tvs.push_back(shift(tv0, {1}));

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  int smem_block_factor = 32;

  tv_out->split(0, smem_block_factor);
  // tv_out->axis(-1)->parallelize(ParallelType::Unswitch);

  auto tv0_cache = tv0->cache_after();

  tv0->computeAt(tv_out, 1);

  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  tv0_cache->setMemoryType(MemoryType::Shared);
  tv_out->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = (t0 + shift(t0, {-1}) + shift(t0, {1})) / 3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift5ptStencilParallel_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 5-pt stencil
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  std::vector<std::vector<int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  std::vector<TensorView*> tvs;
  for (const auto& offset : offsets) {
    tvs.push_back(shift(tv0, offset));
  }

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  int smem_block_factor = 32;

  tv_out->split(-1, smem_block_factor);
  tv_out->split(0, smem_block_factor);

  tv_out->reorder({{1, 2}, {2, 1}});

  auto tv0_cache = tv0->cache_after();

  tv0->computeAt(tv_out, 2);

  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  tv_out->axis(-1)->parallelize(ParallelType::TIDx);
  tv_out->axis(-2)->parallelize(ParallelType::TIDy);
  tv_out->axis(-3)->parallelize(ParallelType::BIDx);
  tv_out->axis(-4)->parallelize(ParallelType::BIDy);

  tv0_cache->setMemoryType(MemoryType::Shared);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-2)->parallelize(ParallelType::TIDy);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = t0;
  for (const auto& offset : offsets) {
    ref = ref + shift(t0, offset);
  }
  ref = ref / int(offsets.size() + 1);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftMerge1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {-1, 1});
  fusion.addOutput(tv2);

  int split_factor = 4;

  tv2->split(-1, split_factor);
  tv2->split(0, split_factor);
  tv2->reorder({{1, 2}, {2, 1}});
  tv2->merge(2, 3);

  tv0->computeAt(tv2, 2);

  // t1 allocation: (split_factor + 1) * (split_factor + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor && rhs_value == 1);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {-1, 1});
  auto ref = t2;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftMerge2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1});
  auto tv3 = shift(tv1, {-1, 1});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  int split_factor = 4;

  tv4->split(-1, split_factor);
  tv4->split(0, split_factor);
  tv4->reorder({{1, 2}, {2, 1}});
  tv4->merge(2, 3);

  tv0->computeAt(tv4, -2);

  // t1 allocation: (split_factor + 2) * (split_factor + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor && rhs_value == 2);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t1, {-1, 1});
  auto t4 = t2 + t3;

  TORCH_CHECK(t4.allclose(outputs[0]));
}

TEST(NVFuserTest, FusionShiftGlobal_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {0, 1});
  auto tv3 = shift(tv1, {-1, 0});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv1->split(-1, 4);
  tv2->split(-1, 8);
  tv3->split(-1, 2);
  tv4->split(-1, 3);

  tv1->merge(-2, -1);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);

  // t1 allocation: (t1.size[0] + 1) * (t1.size[1] + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          TORCH_CHECK(def != nullptr && def->operation() == BinaryOpType::Add);
          TORCH_CHECK(def->as<kir::BinaryOp>()->lhs()->isA<kir::NamedScalar>());
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(rhs_value == 1);
        }
      }
    }
  }

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {0, 1});
  auto t3 = shift(t1, {-1, 0});
  auto t4 = t2 + t3;
  auto ref = t4;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftDoubleSplitMerge1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(2));
  auto tv3 = shift(tv2, {0, 1});
  fusion.addOutput(tv3);

  int split_factor1 = 8;
  int split_factor2 = 4;

  tv3->split(-1, split_factor1);

  tv0->computeAt(tv3, -2);

  tv1->split(-1, split_factor2);
  tv1->merge(-2, -1);

  // t1 and t2 allocation: (split_factor1 + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 1);
        auto def =
            dynamic_cast<kir::BinaryOp*>(alloc->shape().at(0)->definition());
        auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
        TORCH_CHECK(lhs != nullptr && lhs->isConst());
        int lhs_value = *lhs->value();
        auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
        TORCH_CHECK(rhs != nullptr && rhs->isConst());
        int rhs_value = *rhs->value();
        TORCH_CHECK(lhs_value == split_factor1 && rhs_value == 1);
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 3;
  auto ref = shift(t1, {0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftDoubleSplitMerge2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(2));
  auto tv3 = shift(tv2, {1, 1});
  fusion.addOutput(tv3);

  auto out = tv3;

  int split_factor1 = 32;
  int split_factor2 = 4;

  out->split(-1, split_factor1);
  out->split(-1, split_factor2);
  out->split(0, split_factor1);
  out->split(1, split_factor2);
  out->reorder({{3, 1}, {1, 2}, {4, 3}, {2, 4}});
  out->merge(2, 3);
  out->merge(2, 3);
  out->merge(2, 3);
  out->merge(0, 1);

  TransformPropagator::from(out);

  tv0->computeAt(out, 1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out, {tv1, tv2});

  for (auto tv : {tv1, tv2}) {
    tv->setMemoryType(MemoryType::Shared);
  }

  // t1 and t2 allocation: (split_factor1 + 1) * (split_factor1 + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor1 && rhs_value == 1);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = shift(t0 + 1 + 2, {1, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift5ptStencilParallel1DThreadBlock_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 5-pt stencil
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  std::vector<std::vector<int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  std::vector<TensorView*> tvs;
  for (const auto& offset : offsets) {
    tvs.push_back(shift(tv0, offset));
  }

  auto tv_out = tv0;

  for (auto tv : tvs) {
    tv_out = add(tv_out, tv);
  }

  tv_out = div(tv_out, new Double(tvs.size() + 1));

  fusion.addOutput(tv_out);

  std::vector<int> split_factor({4, 32});

  tv_out->split(-1, split_factor[1]);
  tv_out->split(0, split_factor[0]);
  tv_out->reorder({{1, 2}, {2, 1}});

  auto tv0_cache = tv0->cache_after();

  // Merge the inner-most two axes and create
  // a 1D thread block of split_factor1*split_factor2 threads
  tv_out->merge(-2, -1);

  tv0->computeAt(tv_out, 2);

  // Inline completely except for the cache
  for (auto tv : tvs) {
    tv->computeAt(tv_out, -1);
  }

  tv0_cache->merge(-2, -1);

  tv_out->axis(-1)->parallelize(ParallelType::TIDx);
  tv_out->axis(1)->parallelize(ParallelType::BIDx);
  tv_out->axis(0)->parallelize(ParallelType::BIDy);

  tv0_cache->setMemoryType(MemoryType::Shared);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);

  // cache allocation: (split_factor1 + 2) * (split_factor2 + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == tv0_cache->name()) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor[i] && rhs_value == 2);
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = t0;
  for (const auto& offset : offsets) {
    ref = ref + shift(t0, offset);
  }
  ref = ref / int(offsets.size() + 1);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftChain1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = shift(tv0, {0, 1});
  auto tv2 = shift(tv1, {0, 1});
  fusion.addOutput(tv2);

  int split_factor = 4;
  tv2->split(-1, split_factor);

  tv0->computeAt(tv2, -2);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = shift(shift(t0, {0, 1}), {0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftChain2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = shift(tv0, {0, 1});
  auto tv2 = shift(tv1, {0, -1});
  fusion.addOutput(tv2);

  tv2->split(-1, 4);

  tv0->computeAt(tv2, -2);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto ref = shift(shift(t0, {0, 1}), {0, -1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftChain3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {0, 1});
  auto tv3 = shift(tv2, {0, 1});
  fusion.addOutput(tv3);

  int split_factor = 4;
  tv3->split(-1, split_factor);

  tv0->computeAt(tv3, -2);

  // Halo size of tv1 is 2 as it needs to account for both of the two
  // shift operations , while that of tv2 is still just 1

  // tv1: (split_factor + 2)
  // tv2: (split_factor + 1)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 1);
        for (int i = 0; i < 1; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor);
          if (tensor_name == 1) {
            TORCH_CHECK(rhs_value == 2);
          } else if (tensor_name == 2) {
            TORCH_CHECK(rhs_value == 1);
          }
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {0, 1});
  auto t3 = shift(t2, {0, 1});
  auto ref = t3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftChain4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = shift(tv0, {1, -1});
  auto tv2 = shift(tv1, {2, -2});
  auto tv3 = shift(tv2, {3, -3});
  auto tv4 = shift(tv3, {4, -4});
  auto tv_out = tv4;

  fusion.addOutput(tv_out);

  int split_factor = 4;

  tv_out->split(-1, split_factor);
  tv_out->split(0, split_factor);
  tv_out->reorder({{1, 2}, {2, 1}});

  tv0->computeAt(tv_out, 2);

  tv1->merge(-2, -1);
  tv2->merge(-2, -1);
  tv3->merge(-2, -1);

  // tv1: (split_factor + 9) * (split_factor + 9)
  // tv2: (split_factor + 7) * (split_factor + 7)
  // tv3: (split_factor + 4) * (split_factor + 4)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == 1 || tensor_name == 2) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor);
          if (tensor_name == 1) {
            TORCH_CHECK(rhs_value == 9);
          } else if (tensor_name == 2) {
            TORCH_CHECK(rhs_value == 7);
          } else if (tensor_name == 3) {
            TORCH_CHECK(rhs_value == 4);
          }
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = shift(t0, {1, -1});
  auto t2 = shift(t1, {2, -2});
  auto t3 = shift(t2, {3, -3});
  auto t4 = shift(t3, {4, -4});
  auto ref = t4;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShift5ptStencilChain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  std::vector<std::vector<int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  // First stencil: 5pt stencil
  // stencil1 = (tv0 + tv0[+1][0] + tv0[-1][0] + tv0[0][+1] + tv0[0][-1]) / 5
  std::vector<TensorView*> tv_stencil1_shifts;
  for (const auto& offset : offsets) {
    tv_stencil1_shifts.push_back(shift(tv0, offset));
  }

  auto tv_stencil1 = tv0;
  for (auto tv : tv_stencil1_shifts) {
    tv_stencil1 = add(tv_stencil1, tv);
  }

  tv_stencil1 = div(tv_stencil1, new Double(tv_stencil1_shifts.size() + 1));

  // Second stencil: Same 5pt stencil
  std::vector<TensorView*> tv_stencil2_shifts;
  for (const auto& offset : offsets) {
    tv_stencil2_shifts.push_back(shift(tv_stencil1, offset));
  }

  auto tv_stencil2 = tv_stencil1;
  for (auto tv : tv_stencil2_shifts) {
    tv_stencil2 = add(tv_stencil2, tv);
  }

  tv_stencil2 = div(tv_stencil2, new Double(tv_stencil2_shifts.size() + 1));

  auto tv_out = tv_stencil2;

  fusion.addOutput(tv_out);

  auto tv0_cache = tv0->cache_after();

  std::vector<int> split_factor({16, 16});

  tv_out->split(-1, split_factor[1]);
  tv_out->split(0, split_factor[0]);
  tv_out->reorder({{1, 2}, {2, 1}});

  tv0->computeAt(tv_out, 2);

  // Inline completely all inputs to the first stencil output, except for the
  // tv0 cache
  for (auto tv : tv_stencil1_shifts) {
    tv->computeAt(tv_stencil1, -1);
  }

  // Inline completely all inputs to the second stencil output, except
  // for the first stencil output
  for (auto tv : tv_stencil2_shifts) {
    tv->computeAt(tv_stencil2, -1);
  }

  tv_out->axis(1)->parallelize(ParallelType::BIDx);
  tv_out->axis(0)->parallelize(ParallelType::BIDy);

  auto all_values = DependencyCheck::getAllValsBetween(
      {fusion.inputs().begin(), fusion.inputs().end()}, fusion.outputs());
  for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
    tv->axis(-2)->parallelize(ParallelType::TIDy);
  }

  tv0_cache->setMemoryType(MemoryType::Shared);
  tv_stencil1->setMemoryType(MemoryType::Shared);

  // tv0_cache: (split_factor + 4) * (split_factor + 4)
  // tv_stencil1: (split_factor + 2) * (split_factor + 2)
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->irNodes()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node.get())) {
      auto tensor_name = alloc->buffer()->name();
      if (tensor_name == tv0_cache->name() ||
          tensor_name == tv_stencil1->name()) {
        TORCH_CHECK(alloc->shape().size() == 2);
        for (int i = 0; i < 2; ++i) {
          auto def =
              dynamic_cast<kir::BinaryOp*>(alloc->shape().at(i)->definition());
          auto lhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->lhs());
          TORCH_CHECK(lhs != nullptr && lhs->isConst());
          int lhs_value = *lhs->value();
          auto rhs = dynamic_cast<kir::Int*>(def->as<kir::BinaryOp>()->rhs());
          TORCH_CHECK(rhs != nullptr && rhs->isConst());
          int rhs_value = *rhs->value();
          TORCH_CHECK(lhs_value == split_factor[i]);
          if (tensor_name == tv0_cache->name()) {
            TORCH_CHECK(rhs_value == 4);
          } else if (tensor_name == tv_stencil1->name()) {
            TORCH_CHECK(rhs_value == 2);
          }
        }
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto stencil1 = t0;
  for (const auto& offset : offsets) {
    stencil1 = stencil1 + shift(t0, offset);
  }
  stencil1 = stencil1 / int(offsets.size() + 1);
  auto stencil2 = stencil1;
  for (const auto& offset : offsets) {
    stencil2 = stencil2 + shift(stencil1, offset);
  }
  stencil2 = stencil2 / int(offsets.size() + 1);
  auto ref = stencil2;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
