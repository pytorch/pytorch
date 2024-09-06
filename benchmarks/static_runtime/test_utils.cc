// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test_utils.h"

#include <ATen/core/ivalue.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/passes.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/allclose.h>
#endif

#include <memory>
#include <unordered_map>

using namespace torch::jit;
using namespace torch;
using c10::IValue;

namespace torch {
namespace jit {
namespace test {

namespace {

class GraphExecutorWrapper {
 public:
  GraphExecutorWrapper() = default;

  explicit GraphExecutorWrapper(const std::shared_ptr<Graph>& graph)
      : graph_exec_(graph, "") {}

  c10::IValue operator()(const std::vector<c10::IValue>& args) {
    Stack stack(args);
    graph_exec_.run(stack);

    if (stack.size() == 1) {
      return stack[0];
    }
    return c10::ivalue::Tuple::create(stack);
  }

 private:
  GraphExecutor graph_exec_;
};

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
    return torch::jit::StaticModule(
        module_, /* is_frozen */ false, opt, /* sample_inputs */ {});
  }

 private:
  Module module_;
};

class GraphStaticRuntimeContext : public StaticRuntimeTestContext {
 public:
  explicit GraphStaticRuntimeContext(const std::string& source_ir) {
    graph_ = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(source_ir, graph_.get(), vmap);

    graph_exec_ = GraphExecutorWrapper(graph_);
  }

  IValue getExpected(const std::vector<IValue>& args) override {
    return graph_exec_(args);
  }

  StaticModule makeStaticModule(const StaticModuleOptions& opt) const override {
    return StaticModule(graph_, opt, /* sample_inputs */ {});
  }

 private:
  std::shared_ptr<Graph> graph_;
  GraphExecutorWrapper graph_exec_;
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
  for (auto i : c10::irange(l.size())) {
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

} // namespace

void compareResults(
    const IValue& expect,
    const IValue& actual,
    const bool use_allclose,
    const bool use_equalnan) {
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
    auto lhs = expect.toTupleRef().elements();
    auto rhs = actual.toTupleRef().elements();
    ASSERT_TRUE(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
      compareResults(lhs[i], rhs[i]);
    }
  } else if (expect.isList()) {
    EXPECT_TRUE(actual.isList());
    auto lhs = expect.toList();
    auto rhs = actual.toList();
    ASSERT_TRUE(lhs.size() == rhs.size());
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
      ASSERT_FALSE(f == rhs.end());
      compareResults(lh.value(), f->value());
    }
  } else {
    // fall back to the default comparison impl in IValue
    EXPECT_TRUE(expect == actual);
  }
}

at::Tensor getTensor(const at::IValue& ival) {
  if (ival.isTensor()) {
    return ival.toTensor();
  } else if (ival.isTensorList()) {
    auto tensor_vec = ival.toTensorVector();
    TORCH_CHECK(tensor_vec.size() == 1);
    return tensor_vec[0];
  } else if (ival.isTuple()) {
    auto tuple = ival.toTuple();
    auto ivalue_vec = tuple->elements();
    TORCH_CHECK(ivalue_vec.size() == 1);
    return ivalue_vec[0].toTensor();
  } else {
    CAFFE_THROW("Unknown input IValue");
  }
}

Node* getNodeWithKind(const StaticModule& smodule, const std::string& kind) {
  return smodule.findNodeWithKindForTesting(kind);
}

Node* getNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind) {
  const auto symbol = c10::Symbol::fromQualString(kind);
  DepthFirstGraphNodeIterator it(graph);
  for (auto* node = it.next(); node != nullptr; node = it.next()) {
    if (node->kind() == symbol) {
      return node;
    }
  }
  return nullptr;
}

bool hasNodeWithKind(const StaticModule& smodule, const std::string& kind) {
  return getNodeWithKind(smodule, kind) != nullptr;
}

bool hasNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind) {
  return getNodeWithKind(graph, kind) != nullptr;
}

std::shared_ptr<Graph> getGraphFromScript(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  return module.get_method("forward").graph();
}

std::shared_ptr<Graph> getGraphFromIR(const std::string& ir) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(ir, graph.get(), vmap);
  return graph;
}

void compareResultsWithJIT(
    StaticRuntime& runtime,
    const std::shared_ptr<Graph>& graph,
    const std::vector<c10::IValue>& args,
    const bool use_allclose,
    const bool use_equalnan) {
  GraphExecutorWrapper graph_exec(graph);
  auto expected = graph_exec(args);
  auto actual = runtime(args, {});
  runtime.check_for_memory_leak();
  compareResults(expected, actual, use_allclose, use_equalnan);
}

void testStaticRuntime(
    const std::string& source,
    const std::vector<IValue>& args,
    const std::vector<IValue>& args2,
    const bool use_allclose,
    const bool use_equalnan,
    const bool check_resize) {
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
    for (bool manage_output_tensors : {true, false}) {
      for (bool enable_tensorexpr_fusion : {true, false}) {
        if (!enable_out_variant && manage_output_tensors) {
          continue;
        }
        // run static runtime three times
        // 1st run: collect allocation profiles (args)
        // 2nd run: exercise memory planner and resizing with args2
        // 3rd run: run with args again
        StaticModuleOptions opts;
        opts.enable_out_variant = enable_out_variant;
        opts.optimize_memory = enable_out_variant;
        opts.manage_output_tensors = manage_output_tensors;
        opts.enable_tensorexpr_fusion = enable_tensorexpr_fusion;

        auto smodule = test_context->makeStaticModule(opts);
        StaticRuntime runtime(smodule);
        auto actual = runtime(args, {});
        if (actual.isTensor()) {
          EXPECT_GE(smodule.num_nodes(), 2)
              << "If we only have one node, the output of the op we are testing is "
              << "not being managed by the memory planner! A failure here "
              << "can typically be fixed by clone()ing the output of the test script.";
        }
        runtime.check_for_memory_leak();
        // first run
        VLOG(2) << "enable_out_variant: " << enable_out_variant;
        VLOG(2) << "manage_output_tensors: " << manage_output_tensors;
        VLOG(2) << "enable_tensorexpr_fusion: " << enable_tensorexpr_fusion;
        VLOG(2) << "args: " << args;
        VLOG(2) << "args2: " << args2;
        VLOG(2) << "expect: " << expect;
        VLOG(2) << "actual: " << actual;
        compareResults(expect, actual, use_allclose, use_equalnan);
        VLOG(2) << "first run comparison done";
        if (manage_output_tensors) {
          actual = IValue();
          runtime.deallocateOutputTensors();
          runtime.checkOutputTensorMemoryLeaks();
        }

        if (!args2.empty()) {
          auto* memory_planner = runtime.get_memory_planner();
          size_t managed_bytes =
              memory_planner ? memory_planner->total_managed() : 0;

          // Run static runtime again with inputs of a different shape.
          expect = test_context->getExpected(args2);
          actual = runtime(args2, {});
          runtime.check_for_memory_leak();
          VLOG(2) << "comparing with args2";
          compareResults(expect, actual, use_allclose, use_equalnan);
          VLOG(2) << "second run comparison done";
          if (manage_output_tensors) {
            actual = IValue();
            runtime.deallocateOutputTensors();
            runtime.checkOutputTensorMemoryLeaks();
          }

          size_t new_managed_bytes =
              memory_planner ? memory_planner->total_managed() : 0;
          if (check_resize && new_managed_bytes >= 0) {
            EXPECT_GE(new_managed_bytes, managed_bytes);
          }

          // Run static runtime again with an input of the shape observed during
          // the profile run.
          expect = test_context->getExpected(args);
          actual = runtime(args, {});
          runtime.check_for_memory_leak();
          // third run
          VLOG(2) << "comparing third run";
          compareResults(expect, actual, use_allclose, use_equalnan);
          VLOG(2) << "third run comparison done";
          if (manage_output_tensors) {
            actual = IValue();
            runtime.deallocateOutputTensors();
            runtime.checkOutputTensorMemoryLeaks();
          }
        } else {
          // run static runtime again to exercise the memory planner
          // and allocate managed tensors.
          actual = runtime(args, {});
          runtime.check_for_memory_leak();
          VLOG(2) << "comparing second run with same args";
          compareResults(expect, actual, use_allclose, use_equalnan);
          VLOG(2) << "second run comparison done";
          if (manage_output_tensors) {
            actual = IValue();
            runtime.deallocateOutputTensors();
            runtime.checkOutputTensorMemoryLeaks();
          }
          // third run to use the allocated managed tensors.
          actual = runtime(args, {});
          runtime.check_for_memory_leak();
          if (manage_output_tensors) {
            actual = IValue();
            runtime.deallocateOutputTensors();
            runtime.checkOutputTensorMemoryLeaks();
          }
        }
      }
    }
  }

  // make sure inputs were not modified
  VLOG(2) << "Printing out input tensors";
  compareTensorLists(args_tensors, args_copy, use_allclose, use_equalnan);
}

bool hasProcessedNodeWithName(
    torch::jit::StaticModule& smodule,
    const char* name) {
  return smodule.findNodeWithKindForTesting(name) != nullptr;
}

} // namespace test
} // namespace jit
} // namespace torch
