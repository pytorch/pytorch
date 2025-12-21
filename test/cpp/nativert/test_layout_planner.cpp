#include <gtest/gtest.h>

#include <utility>

#define LayoutPlannerTests_TEST_FRIENDS                                  \
  friend class LayoutPlannerCtorTests;                                   \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstruct);                    \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructSymbolicShape);       \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructNoMetadata);          \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructPlanWithOverlap);     \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructPlanNoOverlap);       \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructNoOutVariant);        \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructOutputAlias);         \
  FRIEND_TEST(                                                           \
      LayoutPlannerCtorTests, TestConstructPlanWithMaybeAliasingToCopy); \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructListPackNoUnpack);    \
  FRIEND_TEST(LayoutPlannerCtorTests, TestConstructTensorList);

#include <torch/csrc/autograd/generated/variable_factories.h> // @manual

#include <torch/nativert/executor/Executor.h> // @manual
#include <torch/nativert/executor/SerialGraphExecutor.h> // @manual
#include <torch/nativert/executor/Weights.h> // @manual
#include <torch/nativert/executor/memory/LayoutManager.h> // @manual
#include <torch/nativert/kernels/KernelFactory.h> // @manual
#include <torch/nativert/kernels/KernelHandlerRegistry.h> // @manual
#include <torch/nativert/kernels/KernelRegistry.h> // @manual

using namespace ::testing;

namespace torch::nativert /* must be same as namespace that includes TEST_FRIEND
                             declarations */
{

class LayoutPlannerCtorTests : public testing::Test {
 public:
  void SetUp() override {
    // register static dispatch kernel handler
    register_kernel_handlers();
  }
  void TearDown() override {
    executor_config.reset();
    graph.reset();
    executor.reset();
  }

  void createPlannerForModel(
      const std::string& model,
      const ExecutorConfig& cfg = {},
      const std::unordered_map<std::string, torch::_export::TensorMeta>&
          tensorMeta = {}) {
    executor_config = std::make_unique<ExecutorConfig>(cfg);

    graph = stringToGraph(model);

    if (!tensorMeta.empty()) {
      graph->setTensorValuesMeta(tensorMeta);
    }

    auto kernels = KernelFactory().initializeNodeKernels(
        *graph, nullptr, *executor_config, nullptr);

    auto kernelSchemas = Executor::getKernelSchemas(kernels.nodeKernels);

    planner = std::make_unique<LayoutPlanner>(
        *graph,
        kernelSchemas,
        ExecutionFrame::getPersistentValueMask(*graph),
        executor_config->layoutPlannerSettings);

    frame = std::make_unique<ExecutionFrame>(
        *graph, Weights(graph.get()), *executor_config, planner.get());

    executor = std::make_unique<SerialGraphExecutor>(
        *graph, std::move(kernels.nodeKernels), *executor_config);
  }

  torch::_export::TensorMeta createSymbolicTensorMeta(
      const std::vector<int64_t>& dims,
      std::string device = "cpu",
      torch::_export::ScalarType dtype = torch::_export::ScalarType::FLOAT) {
    torch::_export::TensorMeta out_meta;

    torch::_export::Device d;
    d.set_type(std::move(device));
    out_meta.set_device(d);

    std::vector<torch::_export::SymInt> symvec;
    for (size_t i = 0; i < dims.size(); ++i) {
      torch::_export::SymInt symint;
      torch::_export::SymExpr symexpr;
      symexpr.set_expr_str(std::string("s") + std::to_string(i));
      symint.set_as_expr(symexpr);
      symvec.push_back(symint);
    }

    out_meta.set_sizes(symvec);
    out_meta.set_dtype(dtype);
    out_meta.set_layout(torch::_export::Layout::Strided);

    {
      torch::_export::SymInt i;
      i.set_as_int(0);
      out_meta.set_storage_offset(i);
    }

    return out_meta;
  }

  torch::_export::TensorMeta createTensorMeta(
      const std::vector<int64_t>& dims,
      std::string device = "cpu",
      torch::_export::ScalarType dtype = torch::_export::ScalarType::FLOAT) {
    torch::_export::TensorMeta out_meta;

    torch::_export::Device d;
    d.set_type(std::move(device));
    out_meta.set_device(d);

    std::vector<torch::_export::SymInt> symvec;
    for (const auto dim : dims) {
      torch::_export::SymInt symint;
      symint.set_as_int(dim);
      symvec.push_back(symint);
    }

    out_meta.set_sizes(symvec);
    out_meta.set_dtype(dtype);
    out_meta.set_layout(torch::_export::Layout::Strided);

    {
      torch::_export::SymInt i;
      i.set_as_int(0);
      out_meta.set_storage_offset(i);
    }

    return out_meta;
  }

 protected:
  std::unique_ptr<Graph> graph;
  std::unique_ptr<ExecutionFrame> frame;
  std::unique_ptr<SerialGraphExecutor> executor;
  std::unique_ptr<LayoutPlanner> planner;
  std::unique_ptr<ExecutorConfig> executor_config;
};

namespace {
ExecutorConfig create_enabled_executor_config() {
  ExecutorConfig cfg;
  cfg.enableStaticCPUKernels = true;
  cfg.layoutPlannerSettings =
      LayoutPlannerSettings()
          .setAlgorithmType(LayoutPlannerAlgorithmType::GreedyBySize)
          .setEnabled(true)
          .setLayoutManagerSettings(
              LayoutManagerSettings().setDeallocateBetweenRequests(false));
  return cfg;
};
} // namespace

TEST_F(LayoutPlannerCtorTests, TestConstructOutputAlias) {
  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
  return (%out_t))";

  createPlannerForModel(model, create_enabled_executor_config());
  // no outputs
  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructNoOutVariant) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createTensorMeta({10, 10, 10})}};

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";

  auto executor_config = create_enabled_executor_config();
  executor_config.enableStaticCPUKernels = false;

  createPlannerForModel(model, executor_config, meta);
  // no out variant (static dispatch disabled)
  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructTensorList) {
  auto model = R"(
    graph(%y0, %y1):
      %out_t0 = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %out_t1 = torch.ops.aten.matmul.default(self=%y0, other=%y1)

      %l[] = prim.ListPack(l0=%out_t0, l1=%out_t1)
      %x0, %x1 = prim.ListUnpack(self=%l)

      %res0 = torch.ops.aten.clone.default(self=%x0, memory_format=None)
      %res1 = torch.ops.aten.clone.default(self=%x1, memory_format=None)
  return (%res0, %res1))";

  createPlannerForModel(model, create_enabled_executor_config(), {});

  EXPECT_EQ(planner->get_planned_values().size(), 2);

  auto& out_t0_lifetime = planner->planned_allocation_specs_[0].lifetime;
  auto& out_t1_lifetime = planner->planned_allocation_specs_[1].lifetime;

  EXPECT_EQ(
      std::abs(
          static_cast<int64_t>(out_t0_lifetime.start) -
          static_cast<int64_t>(out_t1_lifetime.start)),
      1);
  EXPECT_EQ(
      std::abs(
          static_cast<int64_t>(out_t0_lifetime.end) -
          static_cast<int64_t>(out_t1_lifetime.end)),
      1);
}

TEST_F(LayoutPlannerCtorTests, TestConstructListPackNoUnpack) {
  auto model = R"(graph(%weight1, %weight2):
%weight1_plannable = torch.ops.aten.clone.default(self=%weight1, memory_format=None)
%weights_list[] = prim.ListPack(l0=%weight1_plannable, l1=%weight2)
%weights_cat = torch.ops.aten.cat.default(tensors=%weights_list, dim=0)
return (%weights_cat)
)";

  createPlannerForModel(model, create_enabled_executor_config(), {});

  auto& weight1_plannable_lifetime =
      planner->planned_allocation_specs_[0].lifetime;
  EXPECT_EQ(weight1_plannable_lifetime.start, 1);
  EXPECT_EQ(weight1_plannable_lifetime.end, 3);
}

TEST_F(LayoutPlannerCtorTests, TestConstructReturnTensorListValues) {
  auto model = R"(
    graph(%y0, %y1):
      %out_t0 = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %out_t1 = torch.ops.aten.matmul.default(self=%y0, other=%y1)

      %l[] = prim.ListPack(l0=%out_t0, l1=%out_t1)
      %x0, %x1 = prim.ListUnpack(self=%l)
  return (%x0, %x1))";
  createPlannerForModel(model, create_enabled_executor_config(), {});

  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructInputTensorList) {
  auto model = R"(
    graph(%y0, %y1):
      %l[] = prim.ListPack(l0=%y0, l1=%y1)
      %x0, %x1 = prim.ListUnpack(self=%l)

      %res0 = torch.ops.aten.clone.default(self=%x0, memory_format=None)
      %res1 = torch.ops.aten.clone.default(self=%x1, memory_format=None)
  return (%res0, %res1))";
  createPlannerForModel(model, create_enabled_executor_config(), {});

  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructReturnTensorList) {
  auto model = R"(
    graph(%y0, %y1):
      %y0_clone = torch.ops.aten.clone.default(self=%y0, memory_format=None)
      %y1_clone = torch.ops.aten.clone.default(self=%y1, memory_format=None)

      %l[] = prim.ListPack(l0=%y0_clone, l1=%y1_clone)
  return (%l))";
  createPlannerForModel(model, create_enabled_executor_config(), {});

  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructUnsupportedDevice) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createTensorMeta({10, 10, 10})}};

  {
    torch::_export::Device d;
    d.set_type("cuda");
    meta["out_t"].set_device(std::move(d));
  }

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";
  createPlannerForModel(model, create_enabled_executor_config(), meta);

  // not cpu
  EXPECT_EQ(planner->get_planned_values().size(), 0);
}

TEST_F(LayoutPlannerCtorTests, TestConstructNoMetadata) {
  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";

  createPlannerForModel(model, create_enabled_executor_config());
  // no metadata

  planner->create_plan();
  EXPECT_EQ(planner->planned_allocation_specs_.size(), 1);
  EXPECT_EQ(planner->get_planned_values().size(), 1);
  auto& spec = planner->planned_allocation_specs_[0];
  EXPECT_EQ(spec.size, 0);
  EXPECT_EQ(spec.lifetime.start, 1);
  EXPECT_EQ(spec.lifetime.end, 2);
}

TEST_F(LayoutPlannerCtorTests, TestConstructSymbolicShape) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createSymbolicTensorMeta({10, 10, 10})}};

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";

  createPlannerForModel(model, create_enabled_executor_config(), meta);
  EXPECT_EQ(planner->get_planned_values().size(), 1);
  EXPECT_EQ(planner->planned_allocation_specs_.size(), 1);
  EXPECT_EQ(
      planner->planned_allocation_specs_[0].size,
      0 /* haven't populated IValues yet */);
}

TEST_F(LayoutPlannerCtorTests, TestConstruct) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createTensorMeta({10, 10, 10})}};

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";
  createPlannerForModel(model, create_enabled_executor_config(), meta);

  auto& specs = planner->planned_allocation_specs_;

  EXPECT_EQ(specs.size(), 1);
  EXPECT_EQ(specs[0].lifetime.start, 1);
  EXPECT_EQ(specs[0].lifetime.end, 2);

  c10::IValue tensor = c10::IValue(torch::rand({10, 10, 10}));

  executor->execute(*frame, {tensor, tensor});

  // 10 * 10 * 10 * 4 rounded up to the nearest multiple of 64 ==> 64 * 63 =
  // 4032
  auto aligned_size = LayoutManager::get_aligned_nbytes(
      10 * 10 * 10 * at::elementSize(at::ScalarType::Float));
  EXPECT_EQ(specs[0].size, aligned_size);
  EXPECT_EQ(specs[0].size, 4032);

  planner->with_plan([&](const LayoutPlan& plan) {
    EXPECT_EQ(plan.total_size, 4032);
    EXPECT_EQ(plan.allocations.size(), 1);
    EXPECT_EQ(plan.allocations[0].size, 4032);
    EXPECT_EQ(plan.allocations[0].offset, 0);
    return;
  });
}

TEST_F(LayoutPlannerCtorTests, TestConstructPlanNoOverlap) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createTensorMeta({10, 10, 10})},
      {"out2_t", createTensorMeta({10, 10, 10})}};

  auto model = R"(
    graph(%y0, %y1):
      %out1_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res1 = torch.ops.aten.clone.default(self=%out1_t, memory_format=None)
      %out2_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res2 = torch.ops.aten.clone.default(self=%out2_t, memory_format=None)
  return (%res1, %res2))";
  createPlannerForModel(model, create_enabled_executor_config(), meta);

  c10::IValue tensor = c10::IValue(torch::rand({10, 10, 10}));

  executor->execute(*frame, {tensor, tensor});

  planner->with_plan([&](const LayoutPlan& plan) {
    EXPECT_EQ(plan.total_size, 4032);
    EXPECT_EQ(plan.allocations.size(), 2);
    EXPECT_EQ(plan.allocations[0].size, 4032);
    EXPECT_EQ(plan.allocations[0].offset, 0);
    EXPECT_EQ(plan.allocations[1].size, 4032);
    EXPECT_EQ(plan.allocations[1].offset, 0);
    return;
  });
}

TEST_F(LayoutPlannerCtorTests, TestConstructPlanWithOverlap) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"out_t", createTensorMeta({10, 10, 10})},
      {"out2_t", createTensorMeta({10, 10, 10})}};

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %out2_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out2_t, memory_format=None)
      %res1 = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res, %res1))";
  createPlannerForModel(model, create_enabled_executor_config(), meta);

  c10::IValue tensor = c10::IValue(torch::rand({10, 10, 10}));

  executor->execute(*frame, {tensor, tensor});

  planner->with_plan([&](const LayoutPlan& plan) {
    EXPECT_EQ(plan.total_size, 8064);
    EXPECT_EQ(plan.allocations.size(), 2);
    EXPECT_EQ(plan.allocations[0].size, 4032);
    EXPECT_EQ(plan.allocations[0].offset, 0);
    EXPECT_EQ(plan.allocations[1].offset, 4032);
    EXPECT_EQ(plan.allocations[1].size, 4032);
  });
}

TEST_F(LayoutPlannerCtorTests, TestConstructPlanWithMaybeAliasingToCopy) {
  auto model = R"(graph(%input):
          %i1 = torch.ops.aten._to_copy.default(self=%input, dtype=ScalarType::FLOAT, memory_format=None)
          %i2 = torch.ops.aten._to_copy.default(self=%input, dtype=ScalarType::FLOAT, memory_format=None)
          %out_t = torch.ops.aten.matmul.default(self=%i1, other=%i2)
          return (%out_t))";

  createPlannerForModel(model, create_enabled_executor_config());

  c10::IValue tensor = c10::IValue(torch::rand({10, 10, 10}));

  executor->execute(*frame, {tensor});

  // i1 and i2 could alias input, so we should be safe and not plan them
  planner->with_plan([&](const LayoutPlan& plan) {
    EXPECT_EQ(plan.total_size, 0);
    EXPECT_EQ(plan.allocations.size(), 0);
    return;
  });
}

TEST_F(LayoutPlannerCtorTests, TestCreateMultiplePlanners) {
  auto executor_config = create_enabled_executor_config();

  auto model = R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))";

  graph = stringToGraph(model);

  std::vector<std::pair<
      std::unique_ptr<LayoutPlanner>,
      std::vector<std::unique_ptr<OpKernel>>>>
      planners;
  for ([[maybe_unused]] const auto _ : c10::irange(2)) {
    auto kernels = KernelFactory().initializeNodeKernels(
        *graph, nullptr, executor_config, nullptr);
    auto kernelSchemas = Executor::getKernelSchemas(kernels.nodeKernels);
    planners.emplace_back(
        std::make_unique<LayoutPlanner>(
            *graph,
            kernelSchemas,
            ExecutionFrame::getPersistentValueMask(*graph),
            executor_config.layoutPlannerSettings),
        std::move(kernels.nodeKernels));
  }

  c10::IValue tensor = c10::IValue(torch::rand({10, 10, 10}));
  for (auto& [layout_planner, kernels] : planners) {
    ExecutionFrame execution_frame(
        *graph, Weights(graph.get()), executor_config, layout_planner.get());
    SerialGraphExecutor graph_executor(
        *graph, std::move(kernels), executor_config);
    graph_executor.execute(execution_frame, {tensor, tensor});
    layout_planner->with_plan([&](const LayoutPlan& plan) {
      EXPECT_EQ(plan.total_size, 4032);
      EXPECT_EQ(plan.allocations.size(), 1);
      EXPECT_EQ(plan.allocations[0].size, 4032);
      EXPECT_EQ(plan.allocations[0].offset, 0);
      return;
    });
  }
}

} // namespace torch::nativert
