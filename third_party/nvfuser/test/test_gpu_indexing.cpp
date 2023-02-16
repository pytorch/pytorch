#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>

#include <test/cpp/jit/test_utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

#include <torch/torch.h>

namespace nvfuser {

TEST_F(NVFuserTest, FusionIndexing1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto aten_output = t3.add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-2);
  tv4->merge(-2);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto aten_output = t3.add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  TensorView* tv1 = makeConcreteTensor({4, 4, 8});
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv3 = broadcast(tv2, {true, false, false});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 8}, options);
  at::Tensor t1 = at::randn({4, 4, 8}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(3);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv3 = broadcast(tv2, {true, false, true});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv3->merge(0)->merge(0)->split(0, 2)->split(0, 3);
  tv4->merge(0)->merge(0)->split(0, 2)->split(0, 3);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({7}, options);
  at::Tensor t1 = at::randn({5, 7, 11}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.unsqueeze(-1).add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> tensor0_shape{7, 4, 7};
  std::vector<int64_t> tensor1_shape{4, 7};

  TensorView* tv0 = makeSymbolicTensor(tensor0_shape.size());
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(tensor1_shape.size());
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  TensorView* tv3 = sum(tv2, {0, 1});
  fusion.addOutput(tv3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn(tensor0_shape, options);
  at::Tensor input1 = at::randn(tensor1_shape, options);

  std::vector<int64_t> reduction_axes{0, 1};
  auto reduction_params = getReductionHeuristics(&fusion, {input0, input1});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, reduction_params->lparams);
  auto cg_outputs = fe.runFusion({input0, input1}, reduction_params->lparams);

  auto aten_output = input0.add(input1).to(at::kDouble).sum(reduction_axes);

  testValidate(
      &fusion,
      cg_outputs,
      {input0, input1},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      reduction_params->lparams);
}

TEST_F(NVFuserTest, FusionIndexing7_CUDA) {
  // Might be able to use this one without 6 as the heuristics in 6 may change
  // and this test is to cover the same issue.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128);
  tv4->split(0, 4);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing8_CUDA) {
  // Same as 7 but with outer splits instead of inner
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128, false);
  tv4->split(0, 4, false);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing9_CUDA) {
  // Same as 7 but with outer splits instead of inner
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);

  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  const int numel_x = 200;
  const int numel_y = 300;
  const int numel_z = 400;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_y}, options);
  auto at_t3 = at::randn({numel_x, numel_y, numel_z}, options);
  std::vector<c10::IValue> aten_inputs = {at_t0, at_t3};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  auto at_t1 = at_t0.unsqueeze(-1);
  auto at_t2 = at_t1.mul(2.0);

  auto at_t4 = at_t3.add(at_t2);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  std::vector<TensorView*> tvs = {tv0_cache, tv1_cache, tv2, tv3};

  for (auto tv : tvs) {
    tv->split(1, 2, false);
    tv->split(1, 1);
    tv->split(-1, 4);
    // [I0, 2, 1, I1/2/4, 4]
    tv->reorder({{1, 2}, {2, 3}, {3, 1}});
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionIndexing11_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, true, true});
  auto tv4 = add(tv3, tv0);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(1);

  tv4->split(1, 32);
  tv4->split(0, 1);

  tv4->reorder({{2, 1}});

  tv2->computeAt(tv4, 3);

  tv2->setMemoryType(MemoryType::Global);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDy);
  tv4->axis(2)->parallelize(ParallelType::Unswitch);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({w, x, y, z}, options);
  at::Tensor t1 = at::randn({x}, options);

  auto t3 = t1.add(1.0).unsqueeze(-1).unsqueeze(-1);
  auto aten_output = t3.add(t0);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing12_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({9, 5});
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3));
  TensorView* tv4 = sum(tv3, {1});

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  auto tv5 = tv4->rFactor({2});

  tv1->computeAt(tv5, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor aten_input = at::randn({9, 5}, options);

  auto t1 = aten_input.add(1.0);
  auto t2 = t1.add(2.0);
  auto t3 = t1.add(3.0);
  auto t4 = t3.sum(1);

  std::vector<at::Tensor> aten_outputs = {t2, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing13_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Progressively broadcast tensors
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv4 = broadcast(tv3, {false, true});
  TensorView* tv5 = add(tv4, tv1);
  TensorView* tv6 = add(tv5, tv2);

  fusion.addOutput(tv6);

  // Split inner dimension
  tv6->split(1, 4);
  // Merge middle dims with outer dimensions
  tv6->merge(2);
  tv6->merge(0);

  // tv6[I0*I1o, I1i*I2]

  // Compute everything inline
  tv0->computeAt(tv6, -1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9, z = 5;
  at::Tensor t0 = at::randn({y}, options);
  at::Tensor t1 = at::randn({y, z}, options);
  at::Tensor t2 = at::randn({x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto t4 = t3.unsqueeze(-1);
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);

  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};
  std::vector<at::Tensor> aten_outputs = {t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing14_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1, -1});
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [b0, i1]
  auto tv2 = add(tv0, IrBuilder::create<Double>(2.0));

  // [i0, i1]
  auto tv3 = add(tv1, IrBuilder::create<Double>(3.0));

  // [b0, i1]
  auto tv4 = add(tv2, IrBuilder::create<Double>(4.0));

  // [io, i1]
  auto tv5 = add(tv2, tv3);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv0->computeAt(tv4, -1);

  tv3->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9;
  at::Tensor t0 = at::randn({1, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);

  auto t4 = t0 + 2 + 4;
  auto t5 = t0 + 2 + t1 + 3;

  std::vector<c10::IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t4, t5};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

// This excercises indexing with broadcast root axes. Non-broadcast
// axes need to be preferred when propagating index exprs to root
// axes. See, e.g., Index::getConsumerIndex_impl.
TEST_F(NVFuserTest, FusionIndexing15_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->merge(1)->merge(0);
  tv4->split(0, 8);
  tv0->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 10;
  const int by = 20;
  const int bz = 30;
  at::Tensor t0 = at::randn({bx}, options);
  at::Tensor t3 = at::randn({bx, by, bz}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output =
      t0.unsqueeze(-1).expand({bx, by}).unsqueeze(-1).expand({bx, by, bz}) + t3;

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing16_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);

  TensorView* tv1 = makeConcreteTensor({5, 3});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true, false});

  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv2->merge(0);
  tv1->computeAt(tv2, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({5, 3}, options);
  auto t2 = t1.unsqueeze(1);
  auto t3 = t0 + t2;

  std::vector<c10::IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing17_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({4});
  fusion.addInput(tv1);
  auto tv2 = unaryOp(UnaryOpType::Set, tv0);
  auto tv3 = unaryOp(UnaryOpType::Set, tv1);

  auto tv4 = sum(tv2, {0, 2});
  auto tv5 = add(tv4, tv3);
  fusion.addOutput(tv5);

  auto tv6 = broadcast(tv3, {true, false, true});
  auto tv7 = add(tv2, tv6);
  fusion.addOutput(tv7);

  tv2->computeAt(tv4, -1, ComputeAtMode::BestEffort);
  tv3->computeAt(tv7, -1, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({4}, options);

  auto t2 = t0;
  auto t3 = t1;

  std::vector<int64_t> reduction_axes{0, 2};
  auto t4 = t2.sum(reduction_axes);
  auto t5 = add(t4, t3);
  auto t6 = t3.unsqueeze(0).unsqueeze(-1);
  auto t7 = t2.add(t6);

  std::vector<c10::IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t5, t7};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

} // namespace nvfuser
