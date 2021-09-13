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
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
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
        tensor.ndimension(), at::indexing::Slice(0, at::indexing::None));
    if (offset > 0) {
      indices[i] = at::indexing::Slice(0, offset);
    } else {
      indices[i] = at::indexing::Slice(offset, at::indexing::None);
    }
    t.index(indices) = 0;
  }
  return t;
}

// ATen version of tensor shifting
auto gather(
    at::Tensor tensor,
    const std::vector<int>& window_shape,
    const std::vector<std::vector<int>>& pad_width) {
  TORCH_CHECK(
      tensor.ndimension() == window_shape.size(),
      "Invalid window shape: ",
      window_shape,
      ". Size of the window shape is different from the tensor dimension.");
  TORCH_CHECK(
      tensor.ndimension() == pad_width.size(),
      "Invalid pad width: ",
      pad_width,
      ". Size of the pad width is different from the tensor dimension.");
  at::Tensor t = tensor;
  for (size_t i = 0; i < window_shape.size(); ++i) {
    const auto w_size = window_shape[i];
    TORCH_CHECK(w_size != 0);
    const auto& pad = pad_width[i];
    TORCH_CHECK(pad.size() == 2);
    at::Tensor concat_tensor;
    for (int w = 0; w < w_size; ++w) {
      std::vector<int> shift_offsets(t.ndimension(), 0);
      shift_offsets[i] = pad[0] - w;
      auto shifted = shift(t, shift_offsets);
      shifted = shifted.unsqueeze(-1);
      if (w == 0) {
        concat_tensor = shifted;
      } else {
        concat_tensor = at::cat({concat_tensor, shifted}, -1);
      }
    }
    t = concat_tensor;
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
// - The original version uses additional 2 warps to load the halos
//   along the Y dimension. The other 10 warps are used to load a 32x10
//   tile, and all warps will do coalesced loads. No such optimization
//   is done in the fuser version.
TEST(NVFuserTest, FusionHdiff_CUDA) {
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
    inp_neighbors.push_back(shift(inp, offset, false));
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
  auto flx = sub(shift(lap, {0, 0, -1}, false), lap);
  // T14 = T13 - T0
  // T15 = T12 * T14
  // T16 = T15 > 0
  // T17 = T16 ? 0 : T12
  auto flx_cond =
      gt(mul(flx, sub(shift(inp, {0, 0, -1}, false), inp)), new Double(0));
  auto flx0 = where(flx_cond, new Double(0), flx);

  // T18 = shift(T10)
  // T19 = T18 - T10
  auto fly = sub(shift(lap, {0, -1, 0}, false), lap);
  // T20 = shift(T0)
  // T21 = T20 - T0
  // T22 = T19 * T21
  // T23 = T22 > 0
  auto fly_cond =
      gt(mul(fly, sub(shift(inp, {0, -1, 0}, false), inp)), new Double(0));
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
              add(sub(flx0, shift(flx0, {0, 0, 1}, false)),
                  sub(fly0, shift(fly0, {0, 1, 0}, false)))));

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
  out->axis(3)->parallelize(ParallelType::TIDy);
  out->axis(4)->parallelize(ParallelType::TIDx);
  // Apply the same parallelization to all other tensors
  scheduler_utils::parallelizeAllLike(out, ir_utils::allTvs(&fusion));

  // Store intermediate stencil results on smem so that they can be
  // accessed by threads
  for (auto tv : {flx0, fly0, lap}) {
    tv->setMemoryType(MemoryType::Shared);
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
  auto fuser_output = fe.runFusion(inputs)[0];
  // Trim the outer rim
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(0, at::indexing::None),
      at::indexing::Slice(2, -2),
      at::indexing::Slice(2, -2)};
  fuser_output = fuser_output.index(indices);

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
    ref = ref.index(indices);

    testValidate(&fusion, {fuser_output}, inputs, {ref}, __LINE__, __FILE__);
  }
}

TEST(NVFuserTest, FusionHdiffPartialSplit_CUDA) {
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
    inp_neighbors.push_back(shift(inp, offset, false));
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
  auto flx = sub(shift(lap, {0, 0, -1}, false), lap);
  // T14 = T13 - T0
  // T15 = T12 * T14
  // T16 = T15 > 0
  // T17 = T16 ? 0 : T12
  auto flx_cond =
      gt(mul(flx, sub(shift(inp, {0, 0, -1}, false), inp)), new Double(0));
  auto flx0 = where(flx_cond, new Double(0), flx);

  // T18 = shift(T10)
  // T19 = T18 - T10
  auto fly = sub(shift(lap, {0, -1, 0}, false), lap);
  // T20 = shift(T0)
  // T21 = T20 - T0
  // T22 = T19 * T21
  // T23 = T22 > 0
  auto fly_cond =
      gt(mul(fly, sub(shift(inp, {0, -1, 0}, false), inp)), new Double(0));
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
              add(sub(flx0, shift(flx0, {0, 0, 1}, false)),
                  sub(fly0, shift(fly0, {0, 1, 0}, false)))));

  fusion.addOutput(out);

  /////////////////////////////////
  // Scheduling
  /////////////////////////////////

  // Step 1: 2D Tiling

  const int tile_x = 32;
  const int tile_y = 8;

  out->split(-1, tile_x, true, true);
  out->split(-3, tile_y, true, true);
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
  out->axis(3)->parallelize(ParallelType::TIDy);
  out->axis(4)->parallelize(ParallelType::TIDx);
  // Apply the same parallelization to all other tensors
  scheduler_utils::parallelizeAllLike(out, ir_utils::allTvs(&fusion));

  // Store intermediate stencil results on smem so that they can be
  // accessed by threads
  for (auto tv : {flx0, fly0, lap}) {
    tv->setMemoryType(MemoryType::Shared);
  }

  /////////////////////////////////
  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int halo_extent = 2;
  const int numel_x = 64 + halo_extent * 2;
  const int numel_y = 64 + halo_extent * 2;
  const int numel_z = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp_at = at::randn({numel_z, numel_y, numel_x}, options);
  at::Tensor coeff_at = at::randn({numel_z, numel_y, numel_x}, options);
  std::vector<IValue> inputs = {inp_at, coeff_at};
  auto fuser_output = fe.runFusion(inputs)[0];
  // Trim the outer rim
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(0, at::indexing::None),
      at::indexing::Slice(2, -2),
      at::indexing::Slice(2, -2)};
  fuser_output = fuser_output.index(indices);

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
    ref = ref.index(indices);

    testValidate(&fusion, {fuser_output}, inputs, {ref}, __LINE__, __FILE__);
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

TEST(NVFuserTest, FusionGatherPadding1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  const std::vector<int> window_shape = {1, 3};
  const std::vector<std::vector<int>> padding_width = {{0, 0}, {1, 1}};

  auto tv1 = gather(tv0, window_shape, padding_width);

  fusion.addOutput(tv1);

  const int s1 = 11;
  const int s2 = 13;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({s1, s2}, options);

  auto ref = gather(t0, window_shape, padding_width);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0});

  TORCH_CHECK(ref.equal(outputs[0]));
}

TEST(NVFuserTest, FusionGatherPadding2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int> window_shape = {1, 3};
  const std::vector<std::vector<int>> padding_width = {{0, 0}, {1, 1}};

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));

  auto tv2 = gather(tv1, window_shape, padding_width);

  auto tv3 = sum(tv2, {-1});

  fusion.addOutput(tv3);

  tv3->split(1, 32);
  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDy);
  tv3->axis(1)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  const int s1 = 99;
  const int s2 = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({s1, s2}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = gather(t1, window_shape, padding_width);
  auto ref = sum(t2, {-1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionConv2DStatic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Input: [C, H, W]
  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);

  // Weights: [K, C, 3, 3]
  auto w = makeSymbolicTensor(4);
  fusion.addInput(w);

  // Gather a neighbor tile of [3, 3] with padding size of 1 for each
  // side of the spatial dimensions
  auto inp_tile = gather(inp, {1, 3, 3}, {{0, 0}, {1, 1}, {1, 1}});
  // inp_tile: [C, H, W, 1, 3, 3]

  auto inp_bc =
      broadcast(inp_tile, {true, false, false, false, false, false, false});
  auto w_bc = broadcast(w, {false, false, true, true, true, false, false});

  auto inp_times_w = mul(inp_bc, w_bc);

  // Reduce the channel and neighbor tile dimensions
  auto out = sum(inp_times_w, {1, 4, 5, 6});

  fusion.addOutput(out);

  ////////////////////////////////////

  // Cache the input and weight tensors
  auto inp_cache = inp->cache_after();

  // Blocking the spatial dimensions
  const int block_w = 16;
  const int block_h = 4;
  // Blocking the channel dimension
  const int block_c = 8;

  out->split(2, block_h);
  out->split(4, block_w);
  out->reorder({{3, 4}});
  // out: [K, C, Ho, Wo, Hi, Wi, 1, 3, 3]

  out->split(1, block_c);
  // out: [K, Co, Ci, Ho, Wo, Hi, Wi, 1, 3, 3]

  auto out_rf = out->rFactor({1, -3, -2, -1});
  // out_rf: [K, rCo, Ci, Ho, Wo, Hi, Wi, 1, 3, 3]
  // out_rf: [K, Ci, Ho, Wo, Hi, Wi]

  // Create a [block_x, block_y] tile on smem
  inp_cache->computeAt(out, 4);
  // inp_cache: [Co, Ho, Wo, Ci, Hi, Wi]
  inp_cache->setMemoryType(MemoryType::Shared);

  // Move Ci forward
  out_rf->reorder({{-4, -6}, {-5, -4}, {-6, -5}});
  inp_cache->computeAt(out_rf, 5);

  inp_tile->computeAt(out_rf, -1);
  w->computeAt(out_rf, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(1)->parallelize(ParallelType::TIDz);
  out->axis(4)->parallelize(ParallelType::TIDy);
  out->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out, {inp_cache, out_rf});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int dim_h = 99;
  const int dim_w = 101;
  const int dim_c = 10;
  const int dim_f = 20;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_inp = at::randn({dim_c, dim_h, dim_w}, options);
  at::Tensor at_w = at::randn({dim_f, dim_c, 3, 3}, options);
  std::vector<IValue> inputs = {at_inp, at_w};

  auto cg_outputs = fe.runFusion(inputs);

  at_inp = at_inp.unsqueeze(0); // at::conv2d needs the N axis
  auto at_out = at::conv2d(at_inp, at_w, {}, 1, 1);
  at_out = at_out.squeeze(0); // drop the N axis

  testValidate(&fusion, cg_outputs, inputs, {at_out}, __LINE__, __FILE__);
}

// Mostly the same as the static conv test, but the shape of the weights,
// 3x3 in this case, is given dynamically
TEST(NVFuserTest, FusionConv2DDynamic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Input: [C, H, W]
  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);

  // Weights: [K, C, S, T]
  auto w = makeSymbolicTensor(4);
  fusion.addInput(w);

  auto w_h = new Int();
  fusion.addInput(w_h);
  auto w_w = new Int();
  fusion.addInput(w_w);

  auto pad_h = new Int();
  fusion.addInput(pad_h);
  auto pad_w = new Int();
  fusion.addInput(pad_w);

  // Gather a neighbor tile of [w_dim_h, w_dim_w] with padding
  auto inp_tile = gather(
      inp,
      {new Int(1), w_h, w_w},
      {{new Int(0), new Int(0)}, {pad_h, pad_h}, {pad_w, pad_w}});
  // inp_tile: [C, 1, H - w_h + 1, W - w_w + 1, w_h, w_w]

  auto inp_bc =
      broadcast(inp_tile, {true, false, false, false, false, false, false});
  auto w_bc = broadcast(w, {false, false, true, true, true, false, false});

  auto inp_times_w = mul(inp_bc, w_bc);

  // Reduce the channel and neighbor tile dimensions
  auto out = sum(inp_times_w, {1, 4, 5, 6});

  fusion.addOutput(out);

  ////////////////////////////////////
  // Cache the input and weight tensors
  auto inp_cache = inp->cache_after();

  // Blocking the spatial dimensions
  const int block_w = 16;
  const int block_h = 4;
  // Blocking the channel dimension
  const int block_c = 8;

  out->split(2, block_h);
  out->split(4, block_w);
  out->reorder({{3, 4}});
  // out: [K, C, Ho, Wo, Hi, Wi, 1, 3, 3]

  out->split(1, block_c);
  // out: [K, Co, Ci, Ho, Wo, Hi, Wi, 1, 3, 3]

  auto out_rf = out->rFactor({1, -3, -2, -1});
  // out_rf: [K, rCo, Ci, Ho, Wo, Hi, Wi, 1, 3, 3]
  // out_rf: [K, Ci, Ho, Wo, Hi, Wi]

  // Create a [block_x, block_y] tile on smem
  inp_cache->computeAt(out, 4);
  // inp_cache: [Co, Ho, Wo, Ci, Hi, Wi]
  inp_cache->setMemoryType(MemoryType::Shared);

  // Move Ci forward
  out_rf->reorder({{-4, -6}, {-5, -4}, {-6, -5}});
  inp_cache->computeAt(out_rf, 5);

  inp_tile->computeAt(out_rf, -1);
  w->computeAt(out_rf, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(1)->parallelize(ParallelType::TIDz);
  out->axis(4)->parallelize(ParallelType::TIDy);
  out->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out, {inp_cache, out_rf});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int dim_h = 99;
  const int dim_w = 101;
  const int dim_c = 10;
  const int dim_f = 20;
  const int dim_w_h = 3;
  const int dim_w_w = 3;
  const int dim_pad_h = (dim_w_h - 1) / 2;
  const int dim_pad_w = (dim_w_w - 1) / 2;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_inp = at::randn({dim_c, dim_h, dim_w}, options);
  at::Tensor at_w = at::randn({dim_f, dim_c, dim_w_h, dim_w_w}, options);
  std::vector<IValue> inputs = {
      at_inp, at_w, dim_w_h, dim_w_w, dim_pad_h, dim_pad_w};

  auto cg_outputs = fe.runFusion(inputs);

  at_inp = at_inp.unsqueeze(0); // at::conv2d needs the N axis
  auto at_out = at::conv2d(at_inp, at_w, {}, 1, 1);
  at_out = at_out.squeeze(0); // drop the N axis

  testValidate(&fusion, cg_outputs, inputs, {at_out}, __LINE__, __FILE__);
}

// 5x5 followed by 3x3
TEST(NVFuserTest, FusionConv2DDynamicChain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Input: [K1, H, W]
  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);

  // Weights: [K2, K1, S1, T1]
  auto w1 = makeSymbolicTensor(4);
  fusion.addInput(w1);

  // Weights: [K3, K2, S2, T2]
  auto w2 = makeSymbolicTensor(4);
  fusion.addInput(w2);

  auto w1_h = new Int();
  fusion.addInput(w1_h);
  auto w1_w = new Int();
  fusion.addInput(w1_w);

  auto w2_h = new Int();
  fusion.addInput(w2_h);
  auto w2_w = new Int();
  fusion.addInput(w2_w);

  auto pad_h1 = new Int();
  fusion.addInput(pad_h1);
  auto pad_w1 = new Int();
  fusion.addInput(pad_w1);

  auto pad_h2 = new Int();
  fusion.addInput(pad_h2);
  auto pad_w2 = new Int();
  fusion.addInput(pad_w2);

  // Gather a neighbor tile of [w1_h, w1_w] with padding
  auto inp_tile = gather(
      inp,
      {new Int(1), w1_h, w1_w},
      {{new Int(0), new Int(0)}, {pad_h1, pad_h1}, {pad_w1, pad_w1}});
  // inp_tile: [C, 1, H - w1_h + 1, W - w1_w + 1, w1_h, w1_w]

  auto inp_bc =
      broadcast(inp_tile, {true, false, false, false, false, false, false});
  auto w1_bc = broadcast(w1, {false, false, true, true, true, false, false});

  auto inp_times_w1 = mul(inp_bc, w1_bc);

  // Reduce the channel and neighbor tile dimensions
  auto out1 = sum(inp_times_w1, {1, 4, 5, 6});

  // Second conv
  auto out1_tile = gather(
      out1,
      {new Int(1), w2_h, w2_w},
      {{new Int(0), new Int(0)}, {pad_h2, pad_h2}, {pad_w2, pad_w2}});

  auto out1_bc =
      broadcast(out1_tile, {true, false, false, false, false, false, false});
  auto w2_bc = broadcast(w2, {false, false, true, true, true, false, false});

  auto out1_times_w2 = mul(out1_bc, w2_bc);

  auto out2 = sum(out1_times_w2, {1, 4, 5, 6});

  fusion.addOutput(out2);

  ////////////////////////////////////
  // Cache the input and weight tensors
  auto inp_cache = inp->cache_after();

  // Blocking the spatial dimensions
  const int block_w = 16;
  const int block_h = 4;

  out2->split(2, block_h);
  out2->split(4, block_w);
  out2->reorder({{3, 4}});
  // out2: [K3, K2, Ho, Wo, Hi, Wi, 1, 3, 3]

  // Create a [block_x, block_y] tile on smem
  inp_cache->computeAt(out2, 4);
  // inp_cache: [Co, Ho, Wo, Ci, Hi, Wi]
  inp_cache->setMemoryType(MemoryType::Shared);

  // Move Ci forward
  out1->reorder({{5, 3}, {3, 4}, {4, 5}});
  out1->setMemoryType(MemoryType::Shared);

  inp_cache->computeAt(out1, 4);

  inp_tile->computeAt(out1, -1);
  w1->computeAt(out1, -1);

  out1_tile->computeAt(out2, -1);
  w2->computeAt(out2, -1);

  out2->axis(0)->parallelize(ParallelType::BIDx);
  out2->axis(4)->parallelize(ParallelType::TIDy);
  out2->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out2, {inp_cache, out1});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int dim_h = 99;
  const int dim_w = 101;
  const int dim_k1 = 3;
  const int dim_k2 = 5;
  const int dim_k3 = 7;
  const int dim_w1_h = 5;
  const int dim_w1_w = 5;
  const int dim_pad1_h = (dim_w1_h - 1) / 2;
  const int dim_pad1_w = (dim_w1_w - 1) / 2;
  const int dim_w2_h = 3;
  const int dim_w2_w = 3;
  const int dim_pad2_h = (dim_w2_h - 1) / 2;
  const int dim_pad2_w = (dim_w2_w - 1) / 2;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_inp = at::randn({dim_k1, dim_h, dim_w}, options);
  at::Tensor at_w1 = at::randn({dim_k2, dim_k1, dim_w1_h, dim_w1_w}, options);
  at::Tensor at_w2 = at::randn({dim_k3, dim_k2, dim_w2_h, dim_w2_w}, options);
  std::vector<IValue> inputs = {
      at_inp,
      at_w1,
      at_w2,
      dim_w1_h,
      dim_w1_w,
      dim_w2_h,
      dim_w2_w,
      dim_pad1_h,
      dim_pad1_w,
      dim_pad2_h,
      dim_pad2_w};

  auto cg_outputs = fe.runFusion(inputs);

  at_inp = at_inp.unsqueeze(0); // at::conv2d needs the N axis
  auto at_out1 = at::conv2d(at_inp, at_w1, {}, 1, 2);
  auto at_out2 = at::conv2d(at_out1, at_w2, {}, 1, 1);
  at_out2 = at_out2.squeeze(0); // drop the N axis

  testValidate(&fusion, cg_outputs, inputs, {at_out2}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionConv2DStaticEvenSizedWindow_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Input: [C, H, W]
  auto inp = makeSymbolicTensor(3);
  fusion.addInput(inp);

  // Weights: [K, C, 2, 2]
  auto w = makeSymbolicTensor(4);
  fusion.addInput(w);

  // Gather a neighbor tile of [2, 2] with padding size of 1 only for
  // the right side of the spatial dimensions. The left padding is
  // zero so that the output axis stays the same.
  auto inp_tile = gather(inp, {1, 2, 2}, {{0, 0}, {0, 1}, {0, 1}});
  // inp_tile: [C, H, W, 1, 2, 2]

  auto inp_bc =
      broadcast(inp_tile, {true, false, false, false, false, false, false});
  auto w_bc = broadcast(w, {false, false, true, true, true, false, false});

  auto inp_times_w = mul(inp_bc, w_bc);

  // Reduce the channel and neighbor tile dimensions
  auto out = sum(inp_times_w, {1, 4, 5, 6});

  fusion.addOutput(out);

  ////////////////////////////////////

  // Cache the input and weight tensors
  auto inp_cache = inp->cache_after();

  // Blocking the spatial dimensions
  const int block_w = 16;
  const int block_h = 4;
  // Blocking the channel dimension
  const int block_c = 8;

  out->split(2, block_h);
  out->split(4, block_w);
  out->reorder({{3, 4}});
  // out: [K, C, Ho, Wo, Hi, Wi, 1, 2, 2]

  out->split(1, block_c);
  // out: [K, Co, Ci, Ho, Wo, Hi, Wi, 1, 2, 2]

  auto out_rf = out->rFactor({1, -3, -2, -1});
  // out_rf: [K, rCo, Ci, Ho, Wo, Hi, Wi, 1, 2, 2]
  // out_rf: [K, Ci, Ho, Wo, Hi, Wi]

  // Create a [block_x, block_y] tile on smem
  inp_cache->computeAt(out, 4);
  // inp_cache: [Co, Ho, Wo, Ci, Hi, Wi]
  inp_cache->setMemoryType(MemoryType::Shared);

  // Move Ci forward
  out_rf->reorder({{-4, -6}, {-5, -4}, {-6, -5}});
  inp_cache->computeAt(out_rf, 5);

  inp_tile->computeAt(out_rf, -1);
  w->computeAt(out_rf, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(1)->parallelize(ParallelType::TIDz);
  out->axis(4)->parallelize(ParallelType::TIDy);
  out->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out, {inp_cache, out_rf});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int dim_h = 99;
  const int dim_w = 101;
  const int dim_c = 10;
  const int dim_f = 20;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_inp = at::randn({dim_c, dim_h, dim_w}, options);
  at::Tensor at_w = at::randn({dim_f, dim_c, 2, 2}, options);
  std::vector<IValue> inputs = {at_inp, at_w};

  auto cg_outputs = fe.runFusion(inputs);

  at_inp = at_inp.unsqueeze(0); // at::conv2d needs the N axis
  auto at_out = at::conv2d(at_inp, at_w, {}, 1, 1);
  at_out = at_out.squeeze(0); // drop the N axis
  // The shape of the spatial domain is (dim_h+1)x(dim_w+1), whereas
  // the fuser output has dim_h*dim_w. Drop the first elements to make
  // it match with the fuser output.
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(0, at::indexing::None),
      at::indexing::Slice(1, at::indexing::None),
      at::indexing::Slice(1, at::indexing::None)};
  at_out = at_out.index(indices);

  testValidate(&fusion, cg_outputs, inputs, {at_out}, __LINE__, __FILE__);
}

// POC implementation of im2col for 3-by-3 kernels
TEST(NVFuserTest, FusionIm2Col_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Input: [N, C, H, W]
  auto inp = makeSymbolicTensor(4);
  fusion.addInput(inp);

  // Gather a neighbor tile of [3, 3] with padding size of 1 for each
  // side of the spatial dimensions
  auto inp_tile = gather(inp, {1, 1, 3, 3}, {{0, 0}, {0, 0}, {1, 1}, {1, 1}});
  // inp_tile: [N, C, H, W, 1, 1, 3, 3]

  auto inp_col = transpose(inp_tile, {{1, 3}, {2, 1}, {3, 2}});
  // inp_col: [N, H, W, C, 1, 1, 3, 3]

  fusion.addOutput(inp_col);

  ////////////////////////////////////

  // Cache the input tensor
  auto inp_cache = inp->cache_after();

  // Blocking the spatial dimensions
  const int block_w = 16;
  const int block_h = 4;

  auto out = inp_col;

  out->split(1, block_h);
  out->split(3, block_w);
  out->reorder({{2, 3}});
  // out: [N, Ho, Wo, Hi, Wi, C, 1, 1, 3, 3]
  // Move the C axis out of Hi*Wi
  out->reorder({{5, 3}, {3, 4}, {4, 5}});
  // out: [N, Ho, Wo, C, Hi, Wi, 1, 1, 3, 3]

  // Create a [block_x, block_y] tile on smem
  inp_cache->computeAt(out, 4);
  inp_cache->setMemoryType(MemoryType::Shared);
  // Fully inline inp_tile
  inp_tile->computeAt(out, -1);

  out->axis(0)->parallelize(ParallelType::BIDz);
  out->axis(1)->parallelize(ParallelType::BIDy);
  out->axis(2)->parallelize(ParallelType::BIDx);
  out->axis(4)->parallelize(ParallelType::TIDy);
  out->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(out, {inp_cache, inp_tile});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int dim_h = 31;
  const int dim_w = 33;
  const int dim_c = 5;
  const int dim_n = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_inp = at::randn({dim_n, dim_c, dim_h, dim_w}, options);
  std::vector<IValue> inputs = {at_inp};

  auto cg_outputs = fe.runFusion(inputs);

  auto at_out = at::im2col(at_inp, {3, 3}, {1, 1}, {1, 1}, {1, 1});

  // at::im2col outputs [N, C*3*3, N*H]
  at_out = at::transpose(at_out, 1, 2);
  at_out = at::reshape(at_out, {dim_n, dim_h, dim_w, dim_c, 1, 1, 3, 3});

  testValidate(&fusion, cg_outputs, inputs, {at_out}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftNoPadding1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, false);
  auto tv3 = shift(tv1, {-1, 1}, false);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sum(tv4, {0, 1});

  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Shared);

  tv5->split(0, 4);
  tv5->split(-1, 8);
  tv5->reorder({{1, 2}});

  TransformPropagator::from(tv5);

  tv2->computeAt(tv5, -1);
  tv3->computeAt(tv5, -1);

  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-2)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv5, ir_utils::allTvs(&fusion));

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t1, {-1, 1});
  auto t4 = t2 + t3;
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, -1), at::indexing::Slice(1, -1)};
  t4 = t4.index(indices);
  auto ref = t4.sum(at::ArrayRef<int64_t>{0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Split and merge
TEST(NVFuserTest, FusionShiftNoPadding2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, false);
  auto tv3 = shift(tv1, {-1, 1}, false);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sum(tv4, {0, 1});

  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Shared);

  tv5->split(0, 4);
  tv5->split(-1, 8);
  tv5->reorder({{1, 2}});
  tv5->merge(-2, -1);

  TransformPropagator::from(tv5);

  tv2->computeAt(tv5, -1);
  tv3->computeAt(tv5, -1);

  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5, ir_utils::allTvs(&fusion));

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t1, {-1, 1});
  auto t4 = t2 + t3;
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, -1), at::indexing::Slice(1, -1)};
  t4 = t4.index(indices);
  auto ref = t4.sum(at::ArrayRef<int64_t>{0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Split and merge, then welford
TEST(NVFuserTest, FusionShiftNoPadding3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, false);
  auto tv3 = shift(tv1, {-1, 1}, false);
  auto tv4 = add(tv2, tv3);
  auto tvs = Welford(tv4, {0, 1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;

  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv1->setMemoryType(MemoryType::Shared);

  tv_avg->split(0, 4);
  tv_avg->split(-1, 8);
  tv_avg->reorder({{1, 2}});
  tv_avg->merge(-2, -1);

  TransformPropagator::from(tv_avg);

  tv2->computeAt(tv_avg, -1);
  tv3->computeAt(tv_avg, -1);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv_avg, ir_utils::allTvs(&fusion));

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);
  outputs[1] /= (numel_x - 2) * (numel_y - 2);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t1, {-1, 1});
  auto t4 = t2 + t3;
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, -1), at::indexing::Slice(1, -1)};
  t4 = t4.index(indices);
  auto ref_avg = t4.mean(at::ArrayRef<int64_t>{0, 1});
  auto ref_M2 = t4.var(at::ArrayRef<int64_t>{0, 1}, false);
  auto ref_N = at::ones({}, options_int) * (numel_x - 2) * (numel_y - 2);

  testValidate(
      &fusion, outputs, inputs, {ref_avg, ref_M2, ref_N}, __LINE__, __FILE__);
}

// Shift indexing and predication with contiguous merge
TEST(NVFuserTest, FusionShiftNoPaddingContigMerge_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, true);
  auto tv3 = shift(tv1, {-1, 1}, false);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv2->merge(0);
  tv3->merge(0);
  tv4->merge(0);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 9;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, -1), at::indexing::Slice(1, -1)};

  auto fuser_out = outputs[0].index(indices);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t1, {-1, 1});
  auto ref = t2 + t3;

  ref = ref.index(indices);

  testValidate(&fusion, {fuser_out}, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionShiftNoPaddingChain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, false);
  auto tv3 = shift(tv2, {1, -1}, false);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv4->split(0, 4);
  tv4->split(-1, 8);
  tv4->reorder({{1, 2}});

  tv1->computeAt(tv4, 2);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-2)->parallelize(ParallelType::TIDy);

  tv4->axis(0)->parallelize(ParallelType::BIDy);
  tv4->axis(1)->parallelize(ParallelType::BIDx);

  scheduler_utils::parallelizeAllLike(tv4, {tv1, tv2, tv3});

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  int numel_x = 99;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0 + 1;
  auto t2 = shift(t1, {1, -1});
  auto t3 = shift(t2, {1, -1});
  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(2, at::indexing::None), at::indexing::Slice(0, -2)};
  t3 = t3.index(indices);
  auto ref = t3.sum(at::ArrayRef<int64_t>{0, 1});

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Rfactor is not allowed with partial domains
TEST(NVFuserTest, FusionShiftNoPaddingRfactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1, -1}, false);
  auto tv3 = sum(tv2, {0, 1});
  fusion.addOutput(tv3);

  tv3->split(0, 4);
  tv3->split(-1, 8);
  tv3->reorder({{1, 2}});

  ASSERT_ANY_THROW(tv3->rFactor({-2}));
}

TEST(NVFuserTest, FusionPartialSplit1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  // [I]
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(0));
  // [I]
  auto tv2 = shift(tv1, {1}, false);
  // [1:I]
  auto tv3 = shift(tv1, {-1}, false);
  // [0:I-1]
  auto tv4 = add(tv2, tv3);
  // [1:I-1]
  fusion.addOutput(tv4);

  // Partial split of tv4. Split only the valid range, which is
  // [1:-1].
  tv4->split(0, 8, true, true);
  // [(I-2)/8, 8]

  // Propagates the partial split back to tv1. This means that all of
  // the other tensors are also shaped as [(I-2)/8, 8], which appears
  // to mean only the sub region of ((I-2)/8 * 8) is
  // computed for tv1, tv2 and tv3. It's fine for the tv2 and tv3
  // tensors as only that sub region is used by tv4. It's also fine
  // for tv1 since it has halo of size one at each side, so the whole
  // region is actually calculated for tv1.
  tv1->computeAt(tv4, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-2)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(tv4, {tv1, tv2, tv3});

  tv1->setMemoryType(MemoryType::Shared);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  // gridDim.x is ceilDiv(numel_x - 2, 8), not ceilDiv(numel_x, 8),
  // so it's going to be just 2 rather than 3.
  const int numel_x = 18;

  ExpressionEvaluator evaluator(&fusion);
  auto root_extent = tv4->getRootDomain()[0]->extent();
  evaluator.bind(root_extent, numel_x);
  auto extent_eval = evaluator.evaluate(tv4->axis(0)->extent());
  TORCH_CHECK(
      extent_eval.has_value(),
      "Invalid evaluation of outer domain extent of partial split");
  TORCH_CHECK(
      extent_eval.value() == (numel_x - 2) / 8,
      "Invalid extent of outer domain of partial split");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{at::indexing::Slice(1, -1)};

  outputs[0] = outputs[0].index(indices);

  auto ref = (shift(t0, {1}) + shift(t0, {-1})).index(indices);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionPartialSplit2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(0));
  auto tv2 = shift(tv1, {1}, false);
  auto tv3 = shift(tv1, {-1}, false);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto tv5 = add(tv1, new Double(1));
  auto tv6 = add(tv5, new Double(1));
  fusion.addOutput(tv6);

  tv4->split(0, 4, true, true);

  // This causes tv5 and tv6 also to be split with the same partial
  // offsets, however, since they need to be calculated entirely, the
  // resulting code would be invalid. It should be detected as part of
  // initial fusion validation during lowering.
  tv1->computeAt(tv4, 1);

  // Validation should throw an error due to tv5 and tv6.
  ASSERT_ANY_THROW(fusion.printKernel());
}

// 2D version of PartialSplit1
TEST(NVFuserTest, FusionPartialSplit3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(0));
  auto tv2 = shift(tv1, {1, 2}, false);
  auto tv3 = shift(tv1, {-2, -1}, false);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(1, 8, true, true);
  tv4->split(0, 4, true, true);
  tv4->reorder({{1, 2}, {2, 1}});

  tv1->computeAt(tv4, 2);

  tv4->axis(0)->parallelize(ParallelType::BIDy);
  tv4->axis(1)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::TIDy);
  tv4->axis(3)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4, {tv1, tv2, tv3});

  tv1->setMemoryType(MemoryType::Shared);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  const int numel_x = 32 + 3;
  const int numel_y = 32 + 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, -2), at::indexing::Slice(2, -1)};

  outputs[0] = outputs[0].index(indices);

  auto ref = (shift(t0, {1, 2}) + shift(t0, {-2, -1})).index(indices);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Almost same fusion with Shift5ptStencilChain but non-padded shift
// and partial split.
TEST(NVFuserTest, FusionPartialSplit4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  std::vector<std::vector<int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  // First stencil: 5pt stencil
  // stencil1 = (tv0 + tv0[+1][0] + tv0[-1][0] + tv0[0][+1] + tv0[0][-1]) / 5
  std::vector<TensorView*> tv_stencil1_shifts;
  for (const auto& offset : offsets) {
    tv_stencil1_shifts.push_back(shift(tv0, offset, false));
  }

  auto tv_stencil1 = tv0;
  for (auto tv : tv_stencil1_shifts) {
    tv_stencil1 = add(tv_stencil1, tv);
  }

  tv_stencil1 = div(tv_stencil1, new Double(tv_stencil1_shifts.size() + 1));

  // Second stencil: Same 5pt stencil
  std::vector<TensorView*> tv_stencil2_shifts;
  for (const auto& offset : offsets) {
    tv_stencil2_shifts.push_back(shift(tv_stencil1, offset, false));
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

  tv_out->split(-1, split_factor[1], true, true);
  tv_out->split(0, split_factor[0], true, true);
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

  tv_out->axis(0)->parallelize(ParallelType::BIDy);
  tv_out->axis(1)->parallelize(ParallelType::BIDx);
  tv_out->axis(2)->parallelize(ParallelType::TIDy);
  tv_out->axis(3)->parallelize(ParallelType::TIDx);

  auto all_values = DependencyCheck::getAllValsBetween(
      {fusion.inputs().begin(), fusion.inputs().end()}, fusion.outputs());
  for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
    scheduler_utils::parallelizeAllLike(tv_out, {tv});
  }

  tv0_cache->setMemoryType(MemoryType::Shared);
  tv_stencil1->setMemoryType(MemoryType::Shared);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  // Input matrix size is 68x68, and the output is 64x64. Both
  // gridDim.x and gridim.y should be ceilDiv(numel - 4,
  // split_factor), which is 4. If full split is used, the grid
  // dimension would be 5.
  const int numel_x = 64 + 4;
  const int numel_y = 64 + 4;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(2, -2), at::indexing::Slice(2, -2)};

  outputs[0] = outputs[0].index(indices);

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
  auto ref = stencil2.index(indices);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionPartialSplit5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int numel_x = 10;
  const int numel_y = 11;

  // auto tv0 = makeSymbolicTensor(2);
  auto tv0 = makeConcreteTensor({numel_x, numel_y});
  fusion.addInput(tv0);

  auto tv1 = shift(tv0, {0, 1}, false);
  auto tv2 = add(tv1, new Double(1));

  fusion.addOutput(tv2);

  // Partially split tv2 but not tv1. Producer indexing with tv2 as a consumer
  // requires adjustment of the index to account for the difference of split
  // offsets.
  tv2->split(1, 4, true, true);
  tv1->split(1, 4);

  tv1->computeAt(tv2, 1);

  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(0, at::indexing::None),
      at::indexing::Slice(1, at::indexing::None)};

  outputs[0] = outputs[0].index(indices);

  auto ref = (shift(t0, {0, 1}) + 1).index(indices);

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST(NVFuserTest, FusionPartialSplit6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int numel_x = 9;

  auto tv0 = makeConcreteTensor({numel_x});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, new Double(1));
  auto tv2 = shift(tv1, {1}, false);
  auto tv3 = add(tv2, new Double(1));

  fusion.addOutput(tv3);

  // Another mix of partial and non-partial split
  tv1->split(0, 4);
  tv2->split(0, 4, true, true);
  tv3->split(0, 4);

  // Just make it easier for compute-sanitizer to flag invalid memory accesses
  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x}, options);
  std::vector<IValue> inputs = {t0};
  auto outputs = fe.runFusion(inputs);

  std::vector<at::indexing::TensorIndex> indices{
      at::indexing::Slice(1, at::indexing::None)};

  outputs[0] = outputs[0].index(indices);

  auto ref = (shift(t0 + 1, {1}) + 1).index(indices);

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
