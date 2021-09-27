// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "test_utils.h"

#include <ATen/core/ivalue.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <memory>
#include <unordered_map>

using namespace torch::jit;
using namespace torch;
using c10::IValue;

namespace torch {
namespace jit {
namespace test {

namespace {

// Test scripts passed to testStaticRuntime can either be IR or JIT.
// The logic for running the script and producing a corresponding StaticModule
// is a bit different for each case. This logic is encapsulated within concrete
// implementations of this class, and testStaticRuntime is only aware of this
// interface.
class StaticRuntimeTestContext {
 public:
  virtual ~StaticRuntimeTestContext() = default;

  virtual IValue getExpected(const std::vector<IValue>& args) = 0;
  virtual StaticModule makeStaticModule(
      const StaticModuleOptions& opt) const = 0;
};

class ModuleStaticRuntimeTestContext : public StaticRuntimeTestContext {
 public:
  explicit ModuleStaticRuntimeTestContext(const std::string& source_jit)
      : module_("module") {
    module_.define(source_jit);
  }

  IValue getExpected(const std::vector<IValue>& args) override {
    return module_.forward(args);
  }

  StaticModule makeStaticModule(const StaticModuleOptions& opt) const override {
    return torch::jit::StaticModule(module_, /* is_frozen */ false, opt);
  }

 private:
  Module module_;
};

class GraphStaticRuntimeContext : public StaticRuntimeTestContext {
 public:
  explicit GraphStaticRuntimeContext(const std::string& source_ir) {
    graph_ = getGraphFromIR(source_ir);
    graph_exec_ = GraphExecutor(graph_, "");
  }

  IValue getExpected(const std::vector<IValue>& args) override {
    Stack stack(args);
    graph_exec_.run(stack);

    if (stack.size() == 1) {
      return stack[0];
    }
    return c10::ivalue::Tuple::create(stack);
  }

  StaticModule makeStaticModule(const StaticModuleOptions& opt) const override {
    return StaticModule(graph_, opt);
  }

 private:
  std::shared_ptr<Graph> graph_;
  GraphExecutor graph_exec_;
};

std::unique_ptr<StaticRuntimeTestContext> makeTestContext(
    const std::string& source) {
  try {
    return std::make_unique<ModuleStaticRuntimeTestContext>(source);
    // Could not parse as TorchScript, assume it's IR
  } catch (const std::runtime_error&) {
    return std::make_unique<GraphStaticRuntimeContext>(source);
  }
}

void compareTensorLists(
    const std::vector<IValue>& l, /* expects */
    const std::vector<IValue>& r, /* values */
    const bool use_allclose,
    const bool use_equalnan) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    ASSERT_TRUE(l[i].isTensor());
    ASSERT_TRUE(r[i].isTensor());
    VLOG(2) << "expect " << i << ": \n" << l[i] << std::endl;
    VLOG(2) << "output " << i << ": \n" << r[i] << std::endl;
    if (!l[i].toTensor().defined()) {
      EXPECT_TRUE(!r[i].toTensor().defined());
    } else {
      if (use_allclose) {
        EXPECT_TRUE(at::allclose(
            l[i].toTensor(),
            r[i].toTensor(),
            /*rtol*/ 1e-05,
            /*atol*/ 1e-08,
            use_equalnan));
      } else {
        EXPECT_TRUE(l[i].toTensor().equal(r[i].toTensor()));
      }
    }
  }
}

void compareResults(
    const IValue& expect,
    const IValue& actual,
    const bool use_allclose = false,
    const bool use_equalnan = false) {
  if (expect.isTensor()) {
    VLOG(2) << "expect " << expect.toTensor() << std::endl;
    VLOG(2) << "output " << actual.toTensor() << std::endl;
    EXPECT_TRUE(actual.isTensor());
    if (use_allclose) {
      EXPECT_TRUE(at::allclose(
          expect.toTensor(),
          actual.toTensor(),
          /*rtol*/ 1e-05,
          /*atol*/ 1e-08,
          use_equalnan));
    } else {
      EXPECT_TRUE(expect.toTensor().equal(actual.toTensor()));
    }
    return;
  } else if (expect.isTuple()) {
    EXPECT_TRUE(actual.isTuple());
    auto lhs = expect.toTuple()->elements();
    auto rhs = actual.toTuple()->elements();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
      compareResults(lhs[i], rhs[i]);
    }
  } else if (expect.isList()) {
    EXPECT_TRUE(actual.isList());
    auto lhs = expect.toList();
    auto rhs = actual.toList();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
      compareResults(lhs[i], rhs[i]);
    }
  } else if (expect.isGenericDict()) {
    EXPECT_TRUE(actual.isGenericDict());
    auto lhs = expect.toGenericDict();
    auto rhs = actual.toGenericDict();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (auto& lh : lhs) {
      auto f = rhs.find(lh.key());
      EXPECT_FALSE(f == rhs.end());
      compareResults(lh.value(), f->value());
    }
  } else {
    // fall back to the default comparison impl in IValue
    EXPECT_TRUE(expect == actual);
  }
}

} // namespace

std::shared_ptr<Graph> getGraphFromIR(const std::string& ir) {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(ir, graph.get(), vmap);
    return graph;
}

void testStaticRuntime(
    const std::string& source,
    const std::vector<IValue>& args,
    const std::vector<IValue>& args2,
    const bool use_allclose,
    const bool use_equalnan) {
  auto test_context = makeTestContext(source);

  std::vector<IValue> args_tensors, args_copy;
  for (const auto& ival : args) {
    if (ival.isTensor()) {
      args_tensors.emplace_back(ival);
      const at::Tensor& t = ival.toTensor();
      args_copy.emplace_back(t.clone());
    }
  }

  auto expect = test_context->getExpected(args);

  for (bool enable_out_variant : {true, false}) {
    auto smodule = test_context->makeStaticModule(
        {true, enable_out_variant, enable_out_variant});
    auto actual = smodule(args, {});
    if (actual.isTensor()) {
      EXPECT_GE(smodule.nodes().size(), 2)
          << "If we only have one node, the output of the op we are testing is "
          << "not being managed by the memory planner! A failure here "
          << "can typically be fixed by clone()ing the output of the test script.";
    }
    smodule.runtime().check_for_memory_leak();
    // first run
    compareResults(expect, actual, use_allclose, use_equalnan);

    // args2 is used to check for dynamic shapes
    // it also exercises the memory planner
    if (!args2.empty()) {
      expect = test_context->getExpected(args2);
      actual = smodule(args2, {});
      smodule.runtime().check_for_memory_leak();
      // second run
      compareResults(expect, actual, use_allclose, use_equalnan);

      expect = test_context->getExpected(args);
      actual = smodule(args, {});
      smodule.runtime().check_for_memory_leak();
      // third run
      compareResults(expect, actual, use_allclose, use_equalnan);
    } else {
      // run static runtime again to exercise the memory planner
      actual = smodule(args, {});
      smodule.runtime().check_for_memory_leak();
      // second run
      compareResults(expect, actual, use_allclose, use_equalnan);
    }
  }

  // make sure inputs were not modified
  compareTensorLists(args_tensors, args_copy, use_allclose, use_equalnan);
}

} // namespace test
} // namespace jit
} // namespace torch
