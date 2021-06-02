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

// Shift a reduced tensor
TEST(NVFuserTest, FusionShiftReduction1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = shift(tv2, {1});
  fusion.addOutput(tv3);

  tv3->split(0, 4);
  tv0->computeAt(tv3, 1);
  tv0->computeAt(tv2, -1);

  const int numel_x = 9;
  const int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = sum(t1, {1});
  auto t3 = shift(t2, {1});
  auto ref = t3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Parallelized version of FusionShiftReduction1
TEST(NVFuserTest, FusionShiftReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = shift(tv2, {1});
  fusion.addOutput(tv3);

  tv3->split(0, 4);
  tv0->computeAt(tv3, 1);

  tv2->split(-1, 32);
  tv0->computeAt(tv2, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Shared);

  const int numel_x = 201;
  const int numel_y = 301;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = sum(t1, {1});
  auto t3 = shift(t2, {1});
  auto ref = t3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftRfactor1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = shift(tv2, {1});
  fusion.addOutput(tv3);

  tv3->split(0, 4);
  tv0->computeAt(tv3, 1);

  tv2->split(-1, 32);
  auto rf = tv2->rFactor({-2});
  tv0->computeAt(tv2, -1);
  tv0->computeAt(rf, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Shared);

  const int numel_x = 201;
  const int numel_y = 301;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = sum(t1, {1});
  auto t3 = shift(t2, {1});
  auto ref = t3;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftBcast1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = shift(tv2, {0, 1});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  const int numel_x = 9;
  const int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  at::Tensor t1 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t4 = t0.unsqueeze(-1).expand({numel_x, numel_y}) + t1;
  auto ref = t4;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftBcast2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = shift(tv2, {1, 0});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(0, 4);
  tv0->computeAt(tv4, 1);

  const int numel_x = 9;
  const int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  at::Tensor t1 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t2 = t0.unsqueeze(-1).expand({numel_x, numel_y});
  auto t3 = shift(t2, {1, 0});
  auto ref = t3 + t1;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Combine ShiftBcast1 and ShiftBcast2 with parallelization
TEST(NVFuserTest, FusionShiftBcast3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = shift(tv2, {1, 0});
  auto tv4 = shift(tv2, {0, 1});
  auto tv5 = shift(tv2, {-1, -1});
  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv6, tv5);
  auto tv8 = add(tv7, tv1);
  fusion.addOutput(tv8);

  tv8->split(0, 4);
  tv8->split(-1, 4);
  tv0->computeAt(tv8, 1);

  tv8->axis(-1)->parallelize(ParallelType::TIDx);
  for (auto tv : {tv8, tv7, tv6, tv5, tv4, tv3, tv2}) {
    tv->axis(1)->parallelize(ParallelType::TIDy);
  }

  tv2->setMemoryType(MemoryType::Shared);

  const int numel_x = 101;
  const int numel_y = 201;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  at::Tensor t1 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t2 = t0.unsqueeze(-1).expand({numel_x, numel_y});
  auto t3 = shift(t2, {1, 0});
  auto t4 = t2;
  auto t5 = shift(t2, {-1, 0});
  auto ref = t3 + t4 + t5 + t1;

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// See issue #893
TEST(NVFuserTest, FusionShiftSyncPlacement1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv0, new Double(2));
  auto tv3 = add(tv1, tv2);
  auto tv4 = shift(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->split(1, 8);
  tv0->computeAt(tv4, 2);

  tv2->computeAt(tv3, -1);

  tv1->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = t0 + 2;
  auto t3 = add(t1, t2);
  auto t4 = shift(t3, {0, 1});

  testValidate(&fusion, outputs, inputs, {t4}, __LINE__, __FILE__);
}

// See issue #893. Top-level placement.
TEST(NVFuserTest, FusionShiftSyncPlacement2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv0, new Double(2));
  auto tv3 = add(tv1, tv2);
  auto tv4 = shift(tv3, {1});
  fusion.addOutput(tv4);

  tv2->computeAt(tv3, -1);

  tv1->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = t0 + 2;
  auto t3 = add(t1, t2);
  auto t4 = shift(t3, {1});

  testValidate(&fusion, outputs, inputs, {t4}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftSyncPlacement3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = add(tv1, new Double(2));
  auto tv3 = shift(tv2, {1});
  fusion.addOutput(tv3);

  // This doesn't work. syncthreads is needed between tv1 and tv2, but
  // both the loop extent of both tv1 and tv2 has halo, so the loop is
  // not eliminated even though it is parallelized. Moving syncthreads
  // out of the loop would make it placed before tv1, which would make
  // it meaningless.
  // Ideally, an exception should be thrown at this computeAt, but at
  // this point, the fusion is not yet parallelized, nor memory type
  // is set, so this computeAt itself is not an error yet.
  tv1->computeAt(tv2, -1);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // The error should be detected when the fusion is lowered.
  ASSERT_ANY_THROW(fusion.printKernel());
}

// Based on original CUDA provided by Vishal Mehta.
// Major differences with the original version:
// - Boundary processing. We always pad by zero. The original version
//   is only defined for the interior domain.
// - The original version uses additional 2 warps to load the halos
//   along the Y dimension. The other 10 warps are used to load a 32x10
//   tile, and all warps will do coalesced loads. No such optimization
//   is done in the fuser version.
TEST(NVFuserTest, FusionHorizontalDiffusion_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);
  auto coeff = makeSymbolicTensor(3);
  fusion.addInput(coeff);

  std::vector<std::vector<int>> offsets{
      {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

  // T2, T3, T4, T5
  std::vector<TensorView*> inp_neighbors;
  for (const auto& offset : offsets) {
    inp_neighbors.push_back(shift(inp, offset));
  }

  // T8
  TensorView* sum_of_neighbors = nullptr;
  for (auto inp_neighbor : inp_neighbors) {
    if (sum_of_neighbors == nullptr) {
      sum_of_neighbors = inp_neighbor;
    } else {
      sum_of_neighbors = add(sum_of_neighbors, inp_neighbor);
    }
  }

  // T9 = T0 * 4
  // T10 = T9 - T8
  auto lap = sub(mul(inp, new Double(4)), sum_of_neighbors);

  // T11 = shift(T10)
  // T12 = T11 - T10
  auto flx = sub(shift(lap, {0, 0, -1}), lap);
  // T14 = T13 - T0
  // T15 = T12 * T14
  // T16 = T15 > 0
  // T17 = T16 ? 0 : T12
  auto flx_cond = gt(mul(flx, sub(shift(inp, {0, 0, -1}), inp)), new Double(0));
  auto flx0 = where(flx_cond, new Double(0), flx);

  // T18 = shift(T10)
  // T19 = T18 - T10
  auto fly = sub(shift(lap, {0, -1, 0}), lap);
  // T20 = shift(T0)
  // T21 = T20 - T0
  // T22 = T19 * T21
  // T23 = T22 > 0
  auto fly_cond = gt(mul(fly, sub(shift(inp, {0, -1, 0}), inp)), new Double(0));
  // T24 = T23 ? 0 : T19
  auto fly0 = where(fly_cond, new Double(0), fly);

  // T25 = shift(flx0)
  // T26 = T17 - T25
  // T27 = shift(fly0)
  // T28 = T24 - T27
  // T29 = T26 + T28
  // T30 = T1 * T29
  // T31 = T0 - T30
  auto out =
      sub(inp,
          mul(coeff,
              add(sub(flx0, shift(flx0, {0, 0, 1})),
                  sub(fly0, shift(fly0, {0, 1, 0})))));

  fusion.addOutput(out);

  /////////////////////////////////
  // Scheduling
  /////////////////////////////////

  // Step 1: 2D Tiling

  const int tile_x = 32;
  const int tile_y = 8;

  out->split(-1, tile_x);
  out->split(-3, tile_y);
  out->reorder({{-2, -3}});
  inp->computeAt(out, -3);
  coeff->computeAt(out, -3);

  // Step 2: Inlining

  // Inline inputs to lap
  auto lap_vals = DependencyCheck::getAllValsBetween({inp}, {lap});
  for (auto val : ir_utils::filterByType<TensorView>(lap_vals)) {
    if (val != lap && val != inp) {
      val->computeAt(lap, -1);
    }
  }

  // Inline inputs to flx0
  auto flx0_vals = DependencyCheck::getAllValsBetween({lap, inp}, {flx0});
  for (auto val : ir_utils::filterByType<TensorView>(flx0_vals)) {
    if (val != lap && val != flx0 && val != inp) {
      val->computeAt(flx0, -1);
    }
  }

  // Inline inputs to fly0
  auto flxy_vals = DependencyCheck::getAllValsBetween({lap, inp}, {fly0});
  for (auto val : ir_utils::filterByType<TensorView>(flxy_vals)) {
    if (val != lap && val != fly0 && val != inp) {
      val->computeAt(fly0, -1);
    }
  }

  // Inline inputs to out
  auto out_vals = DependencyCheck::getAllValsBetween({flx0, fly0}, {out});
  for (auto val : ir_utils::filterByType<TensorView>(out_vals)) {
    if (val != flx0 && val != fly0 && val != out) {
      val->computeAt(out, -1);
    }
  }

  // Step 3: Parallelization

  // Block parallelization
  out->axis(0)->parallelize(ParallelType::BIDz);
  out->axis(1)->parallelize(ParallelType::BIDy);
  out->axis(2)->parallelize(ParallelType::BIDx);

  // Thread parallelization
  for (auto tv : {out, flx0, fly0, lap}) {
    tv->axis(3)->parallelize(ParallelType::TIDy);
    tv->axis(4)->parallelize(ParallelType::TIDx);
    if (tv != out) {
      tv->setMemoryType(MemoryType::Shared);
    }
  }

  /////////////////////////////////
  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 101;
  int numel_y = 99;
  int numel_z = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp_at = at::randn({numel_z, numel_y, numel_x}, options);
  at::Tensor coeff_at = at::randn({numel_z, numel_y, numel_x}, options);
  std::vector<IValue> inputs = {inp_at, coeff_at};
  auto outputs = fe.runFusion(inputs);

  {
    at::Tensor zeros = at::zeros({numel_z, numel_y, numel_x}, options);
    auto lap = inp_at * 4 -
        (shift(inp_at, {0, 1, 0}) + shift(inp_at, {0, -1, 0}) +
         shift(inp_at, {0, 0, 1}) + shift(inp_at, {0, 0, -1}));
    auto flx = shift(lap, {0, 0, -1}) - lap;
    auto flx_cond = (flx * (shift(inp_at, {0, 0, -1}) - inp_at)) > 0;
    auto flx0 = at::where(flx_cond, zeros, flx);
    auto fly = shift(lap, {0, -1, 0}) - lap;
    auto fly_cond = (fly * (shift(inp_at, {0, -1, 0}) - inp_at)) > 0;
    auto fly0 = at::where(fly_cond, zeros, fly);

    auto ref = inp_at -
        coeff_at *
            ((flx0 - shift(flx0, {0, 0, 1})) + (fly0 - shift(fly0, {0, 1, 0})));

    testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
  }
}

// 3x3 max pooling
TEST(NVFuserTest, FusionMaxPooling_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Format: CHW
  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);

  // 3x3 pooling of the HW spatial domain
  std::vector<std::vector<int>> offsets;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      offsets.push_back({i, j});
    }
  }

  std::vector<TensorView*> inp_tile({inp});
  for (auto offset : offsets) {
    offset.insert(offset.begin(), 0);
    inp_tile.push_back(shift(inp, offset));
  }

  TensorView* max_tensor = nullptr;
  for (auto tv : inp_tile) {
    if (max_tensor == nullptr) {
      max_tensor = tv;
    } else {
      max_tensor = binaryOp(BinaryOpType::Max, max_tensor, tv);
    }
  }

  fusion.addOutput(max_tensor);

  ////////////////////////////////////

  // Cache the input and weight tensors
  auto inp_cache = inp->cache_after();

  // Tiling the spatial domain
  const int tile_x = 32;
  const int tile_y = 8;

  max_tensor->split(-2, tile_y);
  max_tensor->axis(-2)->parallelize(ParallelType::TIDy);
  max_tensor->split(-1, tile_x);
  max_tensor->axis(-1)->parallelize(ParallelType::TIDx);
  max_tensor->reorder({{-3, -2}});

  inp_cache->computeAt(max_tensor, 3);
  inp_cache->axis(-2)->parallelize(ParallelType::TIDy);
  inp_cache->axis(-1)->parallelize(ParallelType::TIDx);
  inp_cache->setMemoryType(MemoryType::Shared);

  auto max_tensor_dep =
      DependencyCheck::getAllValsBetween({inp_cache}, {max_tensor});
  for (auto tv : ir_utils::filterByType<TensorView>(max_tensor_dep)) {
    if (tv == inp_cache || tv == max_tensor) {
      continue;
    }
    tv->computeAt(max_tensor, -1);
  }

  max_tensor->axis(0)->parallelize(ParallelType::BIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int hw = 50;
  const int num_channels = 20;
  const int pooling_window = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_inp = at::randn({num_channels, hw, hw}, options);
  // shift always pads by zero, so if all surrounding values are
  // negative, max pooling would pick a padded value, which isn't the
  // correct behavior. We need to be able to choose the value of
  // padding. In this case, padding by the minimum value would not
  // have this problem. For now, avoid the problem by making sure all
  // values are not negative.
  aten_inp = at::abs(aten_inp);
  std::vector<IValue> inputs = {aten_inp};

  auto outputs = fe.runFusion(inputs);

  auto ref = at::max_pool2d(
      aten_inp, {pooling_window, pooling_window}, {1, 1}, {1, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
