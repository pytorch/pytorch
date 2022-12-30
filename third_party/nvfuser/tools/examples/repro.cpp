TEST_F(NVFuserTest, FusionGeneratedTest_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  {
    auto tv0 = TensorViewBuilder().ndims(1).shape({-1}).contiguity({true}).dtype(DataType::Float).build();
    fusion->addInput(tv0);
    auto tv1 = TensorViewBuilder().ndims(1).shape({-1}).contiguity({true}).dtype(DataType::Float).build();
    fusion->addInput(tv1);
    auto tv2 = TensorViewBuilder().ndims(2).shape({-1, -1}).contiguity({true, true}).dtype(DataType::Half).build();
    fusion->addInput(tv2);
    auto tv3 = expand(broadcast(tv0, {true, true, false}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(768)});
    auto tv4 = expand(broadcast(tv1, {true, true, false}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(768)});
    auto tv5 = view(tv2, {1024, 768}, {1, 1024, 768});
    auto tv6 = castOp(DataType::Float, tv5);
    auto s7 = IrBuilder::create<Double>(0.5);
    auto tv8 = mul(tv6, s7);
    auto s9 = IrBuilder::create<Double>(0.707107);
    auto tv10 = mul(tv6, s9);
    auto tv11 = erf(tv10);
    auto s12 = IrBuilder::create<Double>(1.0);
    auto tv13 = add(tv11, s12);
    auto tv14 = mul(tv8, tv13);
    auto tv15 = castOp(DataType::Half, tv14);
    auto tv16 = castOp(DataType::Float, tv15);
    auto tv17_tv18 = variance_mean(tv16, {2}, 0, false);
    auto tv17 = std::get<0>(tv17_tv18);
    auto tv18 = std::get<1>(tv17_tv18);
    auto tv19 = expand(broadcast(tv17, {false, false, true}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(1)});
    auto tv20 = expand(broadcast(tv18, {false, false, true}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(1)});
    auto s21 = IrBuilder::create<Double>(1e-05);
    auto tv22 = add(tv19, s21);
    auto tv23 = expand(broadcast(tv20, {false, false, false}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(768)});
    auto tv24 = rsqrt(tv22);
    auto tv25 = sub(tv16, tv23);
    auto tv26 = expand(broadcast(tv24, {false, false, false}), {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1024), IrBuilder::create<Int>(768)});
    auto tv27 = mul(tv25, tv26);
    auto tv28 = mul(tv27, tv3);
    auto tv29 = add(tv28, tv4);
    auto tv30 = castOp(DataType::Float, tv29);
    auto tv31 = castOp(DataType::Half, tv30);
    auto tv32 = view(tv31, {1, 1024, 768}, {1024, 768});
    fusion->addOutput(tv5);
    fusion->addOutput(tv16);
    fusion->addOutput(tv20);
    fusion->addOutput(tv24);
    fusion->addOutput(tv32);
  }

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  std::vector<IValue> inputs;
  std::vector<Tensor> outputs;

  {
    auto t0 = at::randn({768}, options);
    inputs.push_back(t0);
    auto t1 = at::randn({768}, options);
    inputs.push_back(t1);
    auto t2 = at::randn({1024, 768}, options).to(ScalarType::Half);
    inputs.push_back(t2);
    auto t3 = t0.unsqueeze(0).unsqueeze(1).expand({1, 1024, 768});
    auto t4 = t1.unsqueeze(0).unsqueeze(1).expand({1, 1024, 768});
    auto t5 = t2.view({1, 1024, 768});
    auto t6 = t5.to(ScalarType::Float);
    auto s7 = 0.5;
    auto t8 = at::mul(t6, s7);
    auto s9 = 0.707107;
    auto t10 = at::mul(t6, s9);
    auto t11 = at::erf(t10);
    auto s12 = 1.0;
    auto t13 = at::add(t11, s12);
    auto t14 = at::mul(t8, t13);
    auto t15 = t14.to(ScalarType::Half);
    auto t16 = t15.to(ScalarType::Float);
    auto t17_t18 = at::var_mean(t16, {2}, 0, false);
    auto t17 = std::get<0>(t17_t18);
    auto t18 = std::get<1>(t17_t18);
    auto t19 = t17.unsqueeze(2).expand({1, 1024, 1});
    auto t20 = t18.unsqueeze(2).expand({1, 1024, 1});
    auto s21 = 1e-05;
    auto t22 = at::add(t19, s21);
    auto t23 = t20.expand({1, 1024, 768});
    auto t24 = at::rsqrt(t22);
    auto t25 = at::sub(t16, t23);
    auto t26 = t24.expand({1, 1024, 768});
    auto t27 = at::mul(t25, t26);
    auto t28 = at::mul(t27, t3);
    auto t29 = at::add(t28, t4);
    auto t30 = t29.to(ScalarType::Float);
    auto t31 = t30.to(ScalarType::Half);
    auto t32 = t31.view({1024, 768});
    outputs.push_back(t5);
    outputs.push_back(t16);
    outputs.push_back(t20);
    outputs.push_back(t24);
    outputs.push_back(t32);
  }

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);
  testValidate(fusion, cg_outputs, inputs, outputs, __LINE__, __FILE__);
}
