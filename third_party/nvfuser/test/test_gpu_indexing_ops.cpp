#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionSelectOpPointwise_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t1 = t0.select(0, idx);
  auto t2 = t0.select(1, idx);
  auto t3 = t0.select(2, idx);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t1, t2, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSelectOpReduction_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0});
  auto tv5 = sum(tv2, {1});
  auto tv6 = sum(tv3, {0, 1});

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t4 = t0.select(0, idx).sum(0);
  auto t5 = t0.select(1, idx).sum(1);
  auto t6 = t0.select(2, idx).sum(IntArrayRef{0, 1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t4, t5, t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSelectOpPersistent_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0}, true);
  auto tv5 = sum(tv2, {1}, true);
  auto tv6 = sum(tv3, {0, 1}, true);

  auto tv7 = add(tv1, tv4);
  auto tv8 = add(tv2, tv5);
  auto tv9 = add(tv3, tv6);

  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t1 = t0.select(0, idx);
  auto t2 = t0.select(1, idx);
  auto t3 = t0.select(2, idx);
  auto t4 = t1.sum(0, true);
  auto t5 = t2.sum(1, true);
  auto t6 = t3.sum(IntArrayRef{0, 1}, true);
  auto t7 = t1 + t4;
  auto t8 = t2 + t5;
  auto t9 = t3 + t6;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t7, t8, t9}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelectSimple_CUDA) {
  for (int i = 0; i < 5; ++i) {
    // fix seed
    std::srand(i);
    at::manual_seed(i);

    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();

    FusionGuard fg(&fusion);
    // dimensionality of the problem
    int nDims = 2;
    int nElem = std::rand() % 1023 + 1;
    int nElem_select = nElem + 115;
    int nFeat = std::rand() % 128 + 1;

    // Set up your input tensor views
    TensorView* tv0 = makeContigTensor(nDims);
    TensorView* tv_idx = makeContigTensor(1, DataType::Int);

    fusion.addInput(tv0);
    fusion.addInput(tv_idx);
    TensorView* tv_sel = index_select(tv0, 0, tv_idx);
    fusion.addOutput(tv_sel);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

    at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
    at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
    at::Tensor output = at::zeros({nElem_select, nFeat}, options);

    std::vector<IValue> aten_inputs = {input0, input_idx};
    auto output_ref = at::index_select(input0, 0, input_idx);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionIndexSelect_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

// Test 1D schedule
// If (n_elems * 2 > device_multiprocessor_count * kThreadX), just use 1D
// scheduler or use 2D scheduler
TEST_F(NVFuserTest, FusionIndexSelect1DSch_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 13;
  int nElem_select = nElem + 1;
  int nFeat = 7;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelect3DTv_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 89;
  int nElem_select = nElem + 35;
  int nFeat0 = 255;
  int nFeat1 = 63;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(27.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat0, nFeat1}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat0, nFeat1}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat0, nFeat1}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 27.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelectCanSch_CUDA) {
  // fix seed
  at::manual_seed(0);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 31;
  int nElem_select = nElem + 15;
  int nFeat = 64;

  // Negative Case I
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_fail;
  FusionGuard fg(&fusion_fail);
  TensorView* tv_pre = makeContigTensor(nDims);
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_fail.addInput(tv_pre);
  fusion_fail.addInput(tv1);
  fusion_fail.addInput(tv0);
  fusion_fail.addInput(tv_idx);
  TensorView* tv_t = mul(tv0, tv_pre);
  TensorView* tv_sel = index_select(tv_t, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion_fail.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input_pre = at::rand_like(input0);

  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat}, options);
  std::vector<IValue> aten_inputs = {input_pre, input1, input0, input_idx};

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion_fail, aten_inputs, true);
  auto sch_fail = SchedulerEntry::canSchedule(
      ScheduleHeuristic::PointWise, &fusion_fail, runtime_info);

  // Negative Case II
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_sum_fail;
  FusionGuard fg_sum(&fusion_sum_fail);
  TensorView* tv_sum_pre = makeContigTensor(nDims);
  TensorView* tv_sum_0 = makeContigTensor(nDims);
  TensorView* tv_sum_1 = makeContigTensor(nDims);
  TensorView* tv_sum_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_sum_fail.addInput(tv_sum_pre);
  fusion_sum_fail.addInput(tv_sum_1);
  fusion_sum_fail.addInput(tv_sum_0);
  fusion_sum_fail.addInput(tv_sum_idx);
  TensorView* tv_sum_t = mul(tv_sum_0, tv_sum_pre);
  TensorView* tv_sum_sel = index_select(tv_sum_t, 0, tv_sum_idx);
  TensorView* tv_sum_2 = mul(tv_sum_1, tv_sum_sel);
  TensorView* tv_sum_add = add(IrBuilder::create<Double>(17.0), tv_sum_2);
  auto tv_sum_3 = sum(tv_sum_add, {1});
  // Register your outputs
  fusion_sum_fail.addOutput(tv_sum_3);
  std::vector<IValue> aten_sum_inputs = {input_pre, input1, input0, input_idx};
  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_sum_info(
      &fusion_sum_fail, aten_sum_inputs, true);
  auto sch_sum_fail = SchedulerEntry::canSchedule(
      ScheduleHeuristic::Reduction, &fusion_sum_fail, runtime_sum_info);

  // Positive  Case I
  Fusion fusion_pass;
  FusionGuard fg_p(&fusion_pass);
  TensorView* tv0_p = makeContigTensor(nDims);
  TensorView* tv1_p = makeContigTensor(nDims);
  TensorView* tv_idx_p = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_pass.addInput(tv1_p);
  fusion_pass.addInput(tv0_p);
  fusion_pass.addInput(tv_idx_p);
  TensorView* tv_sel_p = index_select(tv0_p, 0, tv_idx_p);
  TensorView* tv2_p = mul(tv1_p, tv_sel_p);
  TensorView* tv3_p = add(IrBuilder::create<Double>(17.0), tv2_p);
  // Register your outputs
  fusion_pass.addOutput(tv3_p);
  // Schedule through magic scheduler
  std::vector<IValue> aten_inputs_pass = {input1, input0, input_idx};
  SchedulerRuntimeInfo runtime_info_pass(&fusion_pass, aten_inputs_pass, true);
  auto sch_pass = SchedulerEntry::canSchedule(
      ScheduleHeuristic::PointWise, &fusion_pass, runtime_info_pass);

  TORCH_CHECK(sch_pass == true && sch_fail == false && sch_sum_fail == false);
}

TEST_F(NVFuserTest, FusionIndexSelect_Sum_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv_add = add(IrBuilder::create<Double>(17.0), tv2);
  auto tv3 = sum(tv_add, {1});
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto reduction_params = getReductionHeuristics(&fusion, aten_inputs);
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_add = tv2_ref + 17.0;
  at::Tensor output_ref = output_add.sum({1});

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectIdxTvFuseable_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 23;
  int nElem_select = nElem + 15;
  int nFeat = 32;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  TensorView* tv_idx_pre = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_idx_pre);
  TensorView* tv_idx_ret = add(tv_idx, tv_idx_pre);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx_ret);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  auto input_idx_pre = at::zeros({nElem_select}, options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx, input_idx_pre};
  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelectDim1InRank2_CUDA) {
  for (int i = 0; i < 5; ++i) {
    // fix seed
    std::srand(i);
    at::manual_seed(i);

    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // dimensionality of the problem
    int nDims = 2;
    int nElem = std::rand() % 15 + 1;
    int nElem_select = std::rand() % 10 + 1;
    int nFeat = std::rand() % 7 + 1;

    // Set up your input tensor views
    TensorView* tv0 = makeContigTensor(nDims);
    TensorView* tv1 = makeContigTensor(nDims);
    TensorView* tv_idx = makeContigTensor(1, DataType::Int);
    fusion.addInput(tv1);
    fusion.addInput(tv0);
    fusion.addInput(tv_idx);
    TensorView* tv_sel = index_select(tv0, 1, tv_idx);
    TensorView* tv2 = mul(tv1, tv_sel);
    TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input0 = at::randn({nFeat, nElem}, options); // lookup
    at::Tensor input1 =
        at::randn({nFeat, nElem_select}, options); // output&elemwise
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
    at::Tensor output = at::zeros({nFeat, nElem_select}, options);

    std::vector<IValue> aten_inputs = {input1, input0, input_idx};
    auto tv0_ref = at::index_select(input0, 1, input_idx);
    at::Tensor tv2_ref = tv0_ref * input1;
    at::Tensor output_ref = tv2_ref + 17.0;

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionIndexSelectDim2InRank3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 2, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nFeat0, nFeat1, nElem}, options); // lookup
  at::Tensor input1 =
      at::randn({nFeat0, nFeat1, nElem_select}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nFeat0, nFeat1, nElem_select}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 2, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelectDim1InRank3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nFeat0, nElem, nFeat1}, options); // lookup
  at::Tensor input1 =
      at::randn({nFeat0, nElem_select, nFeat1}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nFeat0, nElem_select, nFeat1}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 1, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexSelectDim2InRank4_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 4;
  int nElem = 4;
  int nElem_select = nElem + 15;
  int nFeat0 = 5;
  int nFeat1 = 7;
  int nFeat2 = 25;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  fusion.addOutput(tv3);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 =
      at::randn({nFeat0, nElem, nFeat1, nFeat2}, options); // lookup
  at::Tensor input1 = at::randn(
      {nFeat0, nElem_select, nFeat1, nFeat2}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor input_idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output =
      at::zeros({nFeat0, nElem_select, nFeat1, nFeat2}, options);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto tv0_ref = at::index_select(input0, 1, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {output_ref}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
