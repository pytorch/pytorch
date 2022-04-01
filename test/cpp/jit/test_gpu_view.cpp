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
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/ir/irparser.h>

#include "test_gpu_validator.h"

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

} // namespace

TEST_F(NVFuserTest, FusionViewDtypeSameSizeOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};

  TensorView* x = makeSymbolicTensor(input_shape.size(), DataType::Float);
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_view = view(x_add_bias, DataType::Int32);
  fusion.addOutput(x_view);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);
  std::vector<IValue> aten_inputs = {at_x, at_bias};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_view = at_x_add_bias.view(at::ScalarType::Int);

  testValidate(&fusion, outputs, aten_inputs, {at_x_view}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewDtypeFailMismatchSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};

  TensorView* x = makeSymbolicTensor(input_shape.size(), DataType::Float);
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  ASSERT_ANY_THROW(view(x_add_bias, DataType::Int));
}

TEST_F(NVFuserTest, FusionViewRfactorExtentReplacement_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion->addInput(tv1);

  auto tv2 = view(tv0, {12, 8}, {4, 3, 8});
  auto tv3 = sum(tv2, {-1});
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  auto tv5 = add(tv1, tv4);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({12, 8}, options);
  auto t1 = at::randn({4, 3}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = at::native::view(t0, {4, 3, 8}).sum({-1}) + 1 + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, 10, 4, 10};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_view = view(x_add_bias, input_shape, output_shape);
  fusion.addOutput(x_view);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);
  std::vector<IValue> aten_inputs = {at_x, at_bias};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_view = at::native::view(at_x_add_bias, output_shape);

  testValidate(&fusion, outputs, aten_inputs, {at_x_view}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionFlattenOutput_CUDA) {
  std::vector<int64_t> input_shape{2, 3, 4, 5};

  auto run_test = [&](int64_t start_dim, int64_t end_dim) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = makeSymbolicTensor(input_shape.size());
    TensorView* bias = makeSymbolicTensor(input_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_add_bias = add(x, bias);
    auto x_view = flatten(x_add_bias, start_dim, end_dim);
    fusion.addOutput(x_view);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(input_shape, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_add_bias = at_x + at_bias;
    auto at_x_view = at::native::flatten(at_x_add_bias, start_dim, end_dim);

    testValidate(
        &fusion, outputs, aten_inputs, {at_x_view}, __LINE__, __FILE__);
  };

  for (int64_t start_dim = 0; start_dim < input_shape.size(); start_dim++) {
    for (int64_t end_dim = start_dim; end_dim < input_shape.size(); end_dim++) {
      run_test(start_dim, end_dim);
    }
  }

  for (int64_t start_dim = -input_shape.size(); start_dim < 0; start_dim++) {
    for (int64_t end_dim = start_dim; end_dim < 0; end_dim++) {
      run_test(start_dim, end_dim);
    }
  }
}

TEST_F(NVFuserTest, FusionViewFailMismatchSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // The number of elements in input and output shapes do not match,
  // so this view transformation is invalid.
  // 2 * 10 * 40 != 2 * 50 * 4 * 10

  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, 50, 4, 10};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  ASSERT_ANY_THROW(view(x_add_bias, input_shape, output_shape));
}

TEST_F(NVFuserTest, FusionViewFailMulitDimInference_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Only one dimension can be inferred in the output shape.
  // Otherwise, the size of the dimensions is ambiguous.
  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, -1, 4, -1};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  ASSERT_ANY_THROW(view(x_add_bias, input_shape, output_shape));
}

void reductionViewAddFusion(
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape,
    bool view_before_reduction) {
  constexpr int kReductionAxis = -1;

  // Drop size for reduction axis from view_shape
  std::vector<int64_t> view_shape;
  {
    const auto kAxis = (kReductionAxis < 0)
        ? (kReductionAxis + input_shape.size())
        : kReductionAxis;
    for (auto i : c10::irange(input_shape.size())) {
      if (view_before_reduction || i != kAxis) {
        view_shape.push_back(input_shape[i]);
      }
    }
  }

  auto bias_shape = (view_before_reduction) ? input_shape : output_shape;
  for (auto has_implicit_broadcast : {false, true}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* x = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    TensorView* bias = (has_implicit_broadcast)
        ? makeConcreteTensor(bias_shape)
        : makeSymbolicTensor(bias_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto tv1 =
        (view_before_reduction) ? add(x, bias) : sum(x, {kReductionAxis});
    auto x_view = view(tv1, view_shape, output_shape);
    auto y = (view_before_reduction) ? sum(x_view, {kReductionAxis})
                                     : add(x_view, bias);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(bias_shape, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

    auto at_tv1 = (view_before_reduction) ? (at_x + at_bias)
                                          : at::sum(at_x, kReductionAxis);
    auto at_x_view = at::native::view(at_tv1, output_shape);
    auto at_y = (view_before_reduction) ? at::sum(at_x_view, kReductionAxis)
                                        : at::add(at_x_view, at_bias);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionViewReductionShmoo_CUDA) {
  typedef std::vector<int64_t> shape;
  typedef std::pair<shape, shape> view_example;

  std::vector<view_example> view_before_examples = {
      {{19, 12, 7, 99}, {19, 3, 2772}},
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 2772}},
      {{3, 17, 80, 1}, {51, 2, 4, 1, 10}},
      {{3, 17, 80, 1, 9}, {51, 2, 4, 1, 10, 9}},
      {{2, 3, 4, 5}, {1, 6, 1, 2, 2, 5, 1}},
      {{22, 22, 2}, {22, 11, 1, 1, 4}},
      {{37, 9, 7, 6, 10}, {333, 2, 2, 3, 35}},
      {{1, 1, 333, 1}, {1, 1, 333, 1}},
      {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1, 8}},
      {{1, 333, 1}, {1, 37, 9, 1}},
      {{1, 333}, {1, 1, 1, 111, 1, 3}},
      {{22, 1, 22, 1}, {484}},
      {{1, 333, 1}, {333}},
      {{1, 27454, 1, 2}, {1, 7844, 1, 7}},
      {{1, 7844, 1, 7}, {1, 27454, 2}}};

  for (auto e : view_before_examples) {
    reductionViewAddFusion(e.first, e.second, true /* view_before_reduction */);
  }

  std::vector<view_example> view_after_examples = {
      {{19, 12, 7, 99}, {19, 3, 28}},
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 28}},
      {{3, 17, 80, 1}, {51, 1, 2, 4, 10}},
      {{3, 17, 80, 1, 9}, {51, 1, 2, 4, 10}},
      {{2, 3, 4, 5}, {1, 6, 1, 2, 2, 1}},
      {{22, 22, 2}, {22, 11, 1, 1, 2}},
      {{37, 9, 7, 6, 10}, {333, 2, 21}},
      {{1, 1, 333, 1}, {1, 1, 333, 1}},
      {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1}},
      {{1, 333, 1}, {1, 37, 9, 1}},
      {{22, 1, 22, 1}, {484}},
      {{1, 333, 1}, {333}},
      {{1, 27454, 1, 2}, {1, 3922, 1, 7}},
      {{1, 7844, 1, 7}, {1, 1961, 4}}};

  for (auto e : view_after_examples) {
    reductionViewAddFusion(
        e.first, e.second, false /* view_before_reduction */);
  }
}

void persistentViewAddFusion(
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape,
    bool view_before_persistent) {
  constexpr int kAxis = -1;

  auto bias_shape = (view_before_persistent) ? input_shape : output_shape;
  for (auto has_implicit_broadcast : {false, true}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* x = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    TensorView* bias = (has_implicit_broadcast)
        ? makeConcreteTensor(bias_shape)
        : makeSymbolicTensor(bias_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto tv1 = (view_before_persistent) ? add(x, bias) : softmax(x, kAxis);
    auto x_view = view(tv1, input_shape, output_shape);
    auto y =
        (view_before_persistent) ? softmax(x_view, kAxis) : add(x_view, bias);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(bias_shape, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

    auto at_tv1 = (view_before_persistent)
        ? (at_x + at_bias)
        : at::_softmax(at_x, kAxis, false /* half_to_float */);
    auto at_x_view = at::native::view(at_tv1, output_shape);
    auto at_y = (view_before_persistent)
        ? at::_softmax(at_x_view, kAxis, false /* half_to_float */)
        : at::add(at_x_view, at_bias);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionViewPersistentShmoo_CUDA) {
  typedef std::vector<int64_t> shape;
  typedef std::pair<shape, shape> view_example;

  std::vector<view_example> view_examples = {
      {{19, 12, 7, 99}, {19, 3, 2772}},
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 2772}},
      {{3, 17, 80, 1}, {51, 2, 4, 1, 10}},
      {{3, 17, 80, 1, 9}, {51, 2, 4, 1, 10, 9}},
      {{2, 3, 4, 5}, {1, 6, 1, 2, 2, 5, 1}},
      {{22, 22, 2}, {22, 11, 1, 1, 4}},
      {{37, 9, 7, 6, 10}, {333, 2, 2, 3, 35}},
      {{1, 1, 333, 1}, {1, 1, 333, 1}},
      {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1, 8}},
      {{1, 333, 1}, {1, 37, 9, 1}},
      {{1, 333}, {1, 1, 1, 111, 1, 3}},
      {{22, 1, 22, 1}, {484}},
      {{1, 333, 1}, {333}},
      {{1, 27454, 1, 2}, {1, 7844, 1, 7}},
      {{1, 7844, 1, 7}, {1, 27454, 2}}};

  for (auto e : view_examples) {
    persistentViewAddFusion(
        e.first, e.second, true /* view_before_persistent */);
  }

  for (auto e : view_examples) {
    persistentViewAddFusion(e.first, e.second, false);
  }
}

void addViewGeluFusion(
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape) {
  for (auto has_implicit_broadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    TensorView* bias = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_add_bias = add(x, bias);
    auto x_view = view(x_add_bias, input_shape, output_shape);
    auto y = gelu(x_view);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(input_shape, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_add_bias = at_x + at_bias;
    auto at_x_view = at::native::view(at_x_add_bias, output_shape);
    auto at_y = at::gelu(at_x_view);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionViewSplit_CUDA) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{2, 4, 10};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionViewBroadcast_CUDA) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{1, 80};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionViewMerge_CUDA) {
  std::vector<int64_t> input_shape{2, 40, 7};
  std::vector<int64_t> output_shape{560};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionViewAllShmoo_CUDA) {
  typedef std::vector<int64_t> shape;
  typedef std::pair<shape, shape> view_example;

  std::vector<view_example> examples = {
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 2772}},
      {{3, 17, 80, 1}, {51, 1, 2, 4, 10}},
      {{3, 17, 80, 1, 9}, {51, 1, 2, 4, 10, 9}},
      {{2, 3, 4, 5}, {1, 6, 1, 2, 2, 5, 1}},
      {{22, 22, 2}, {22, 11, 1, 1, 4}},
      {{37, 9, 7, 6, 10}, {333, 2, 2, 3, 35}},
      {{1, 1, 333, 1}, {1, 1, 333, 1}},
      {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1, 8}},
      {{1, 333, 1}, {1, 37, 9, 1}},
      {{1, 333}, {1, 1, 1, 111, 1, 3}},
      {{22, 1, 22, 1}, {484}},
      {{1, 333, 1}, {333}},
      {{1, 27454, 1, 2}, {1, 7844, 1, 7}},
      {{1, 7844, 1, 7}, {1, 27454, 2}}};

  for (auto e : examples) {
    addViewGeluFusion(e.first, e.second);
  }
}

TEST_F(NVFuserTest, FusionViewInferShmoo_CUDA) {
  typedef std::vector<int64_t> shape;
  typedef std::pair<shape, shape> view_example;

  std::vector<view_example> examples = {
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, -1, 3, 2772}},
      {{3, 17, 80, 1}, {51, 1, 2, 4, -1}},
      {{3, 17, 80, 1, 9}, {-1, 1, 2, 4, 10, 9}},
      {{2, 3, 4, 5}, {1, 6, 1, -1, 2, 5, 1}},
      {{22, 22, 2}, {22, -1, 1, 1, 4}},
      {{37, 9, 7, 6, 10}, {333, 2, -1, 3, 35}},
      {{1, 1, 333, 1}, {1, 1, -1, 1}},
      {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1, -1}},
      {{1, 333, 1}, {1, 37, -1, 1}},
      {{1, 333}, {1, 1, 1, -1, 1, 3}},
      {{22, 1, 22, 1}, {-1}},
      {{1, 333, 1}, {-1}},
      {{1, 27454, 1, 2}, {1, 7844, 1, -1}},
      {{1, 7844, 1, 7}, {1, -1, 2}}};

  for (auto e : examples) {
    addViewGeluFusion(e.first, e.second);
  }
}

void geluViewAddFusion(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> output_shape) {
  for (auto hasImplicitBroadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (hasImplicitBroadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    TensorView* bias = (hasImplicitBroadcast)
        ? makeConcreteTensor(output_shape)
        : makeSymbolicTensor(output_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_gelu = gelu(x);
    auto x_view = view(x_gelu, input_shape, output_shape);
    auto y = add(x_view, bias);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(output_shape, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_gelu = at::gelu(at_x);
    auto at_x_view = at::native::view(at_x_gelu, output_shape);
    auto at_y = at_x_view + at_bias;

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionViewStride_CUDA) {
  typedef std::vector<int64_t> shape;
  typedef std::pair<shape, shape> view_example;

  std::vector<view_example> examples = {
      {{1, 27454, 2}, {1, 7844, 7}},
      {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 2772}},
      {{1, 7844, 1, 7}, {1, 27454, 2}}};

  for (auto e : examples) {
    geluViewAddFusion(e.first, e.second);
  }
}

void geluViewBinaryAddFusion(
    std::vector<int64_t> input_shape1,
    std::vector<int64_t> input_shape2,
    std::vector<int64_t> output_shape) {
  for (auto hasImplicitBroadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (hasImplicitBroadcast)
        ? makeConcreteTensor(input_shape1)
        : makeSymbolicTensor(input_shape1.size());
    TensorView* bias = (hasImplicitBroadcast)
        ? makeConcreteTensor(input_shape2)
        : makeSymbolicTensor(input_shape2.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_gelu = gelu(x);
    auto x_view = view(x_gelu, input_shape1, output_shape);
    auto bias_view = view(bias, input_shape2, output_shape);
    auto y = add(x_view, bias_view);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape1, options);
    at::Tensor at_bias = at::randn(input_shape2, options);
    std::vector<IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_gelu = at::gelu(at_x);
    auto at_x_view = at::native::view(at_x_gelu, output_shape);
    auto at_bias_view = at::native::view(at_bias, output_shape);
    auto at_y = at_x_view + at_bias_view;

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionViewBinary_CUDA) {
  geluViewBinaryAddFusion({27454, 2}, {54908}, {7844, 7});
}

// Repro of issue #1493
TEST_F(NVFuserTest, FusionViewConcreteDomain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = view(tv0, {2, 3}, {6});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  auto tv4 = broadcast(tv3, {true, false});
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(0);
  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({2, 3}, options);
  auto t1 = at::randn({1, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = (at::native::view(t0, {6}) + 1).unsqueeze(0) + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewConcreteDomain2_CUDA) {
  constexpr int kAxis = -1;
  std::vector<int64_t> input_shape = {19, 12, 7, 99};
  std::vector<int64_t> output_shape = {19, 3, 2772};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(output_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto tv1 = softmax(x, kAxis);
  auto x_view = view(tv1, input_shape, output_shape);
  auto y = add(x_view, bias);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(output_shape, options);
  std::vector<IValue> aten_inputs = {at_x, at_bias};

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  auto at_tv1 = at::_softmax(at_x, kAxis, false /* half_to_float */);
  auto at_x_view = at::native::view(at_tv1, output_shape);
  auto at_y = at::add(at_x_view, at_bias);

  testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewConcreteDomain3_CUDA) {
  std::vector<int64_t> input_shape = {14, 12, 8, 100};
  std::vector<int64_t> bcast_shape = {14, 12, 8, 1};
  std::vector<int64_t> other_shape = {14, 100, 96};
  std::vector<int64_t> output_shape = {14, 3, 3200};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* y = makeConcreteTensor(bcast_shape);
  TensorView* z = makeSymbolicTensor(other_shape.size());
  fusion.addInput(x);
  fusion.addInput(y);
  fusion.addInput(z);

  auto tv1 = add(x, y);
  auto tv2 = view(tv1, input_shape, output_shape);
  auto tv3 = view(z, other_shape, output_shape);
  auto output = add(tv2, tv3);
  fusion.addOutput(output);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(bcast_shape, options);
  at::Tensor at_z = at::randn(other_shape, options);
  std::vector<IValue> aten_inputs = {at_x, at_y, at_z};

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  auto at_tv1 = at::add(at_x, at_y);
  auto at_tv2 = at::native::view(at_tv1, output_shape);
  auto at_tv3 = at::native::view(at_z, output_shape);
  auto at_output = at::add(at_tv2, at_tv3);

  testValidate(&fusion, outputs, aten_inputs, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewConcreteDomain4_CUDA) {
  std::vector<int64_t> shape1 = {3, 4, 5};
  std::vector<int64_t> shape2 = {3 * 4 * 5};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(shape1.size() - 1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(shape1.size());
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false, false});
  auto tv3 = add(tv1, tv2);
  auto tv4 = view(tv3, shape1, shape2);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  ComputeAtMap loop_map(&fusion, ComputeAtMap::MappingMode::LOOP);
  ComputeAtMap index_map(&fusion, ComputeAtMap::MappingMode::INDEX);

  TORCH_CHECK(tv5->nDims() == 1);

  // The concrete domain of tv5, which is 1D, in the loop map (or
  // parallel map) needs to be either the domain of tv4 or tv5, both
  // of which have the three concrete root domains of tv1. In other
  // words, it must map with tv4 and tv5 in the index map.
  auto concrete_id = loop_map.getConcreteMappedID(tv5->axis(0));
  TORCH_CHECK(
      index_map.areMapped(concrete_id, tv5->axis(0)),
      "Invalid concrete ID: ",
      concrete_id->toString());
  TORCH_CHECK(
      index_map.areMapped(concrete_id, tv4->axis(0)),
      "Invalid concrete ID: ",
      concrete_id->toString());
}

TEST_F(NVFuserTest, FusionViewConcreteDomain5_CUDA) {
  const std::vector<int64_t> shape1 = {12};
  const std::vector<int64_t> shape2 = {4, 3};
  const std::vector<int64_t> shape3 = {12, 5};
  const std::vector<int64_t> shape4 = {4, 3, 5};

  for (auto order : {true, false}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(1);
    fusion.addInput(tv0);

    auto tv1 = makeSymbolicTensor(2);
    fusion.addInput(tv1);

    auto tv0_cache = set(tv0);

    auto path1 = [&]() {
      auto view_2d = view(tv0_cache, shape1, shape2);
      auto view_2d_copy = set(view_2d);
      fusion.addOutput(view_2d_copy);
      return view_2d_copy;
    };

    auto path2 = [&]() {
      auto tv0_bc = broadcast(tv0_cache, {false, true});
      auto tv0_bc_plus_tv1 = add(tv0_bc, tv1);
      auto view_3d = view(tv0_bc_plus_tv1, shape3, shape4);
      auto view_3d_copy = set(view_3d);
      fusion.addOutput(view_3d_copy);
      return view_3d_copy;
    };

    TensorView* path1_out = nullptr;
    TensorView* path2_out = nullptr;

    if (order) {
      // Fails before #1544. Concrete ID is picked from path1_out, which
      // doesn't have the second root domain of tv1
      path2_out = path2();
      path1_out = path1();
    } else {
      // Works fine
      path1_out = path1();
      path2_out = path2();
    }

    path2_out->merge(-2, -1);
    path2_out->merge(-2, -1);

    tv0->computeAt(path2_out, -1);
    tv1->computeAt(path2_out, -1);

    TORCH_CHECK(path1_out->nDims() == 1);
    TORCH_CHECK(path2_out->nDims() == 1);

    ComputeAtMap par_map(&fusion, ComputeAtMap::MappingMode::PARALLEL);

    // Make sure the two output tensors are mapped. Note both are 1D.
    TORCH_CHECK(par_map.areMapped(path1_out->axis(0), path2_out->axis(0)));

    auto concrete_id = par_map.getConcreteMappedID(path2_out->axis(0));
    TORCH_CHECK(
        path2_out->axis(0) == concrete_id,
        "Incorrect concrete ID: ",
        concrete_id->toString());
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
