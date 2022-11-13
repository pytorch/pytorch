#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <fstream>

#include <unordered_set>

#include <torch/csrc/profiler/events.h>

#include "tools/cxx/Resources.h"

#ifdef EDGE_PROFILER_USE_KINETO
namespace torch {
namespace jit {
namespace mobile {

namespace {
bool checkMetaData(
    const std::string& op_name,
    const std::string& metadata_name,
    const std::string& metadata_val,
    std::ifstream& trace_file) {
  std::string line;
  while (std::getline(trace_file, line)) {
    if (line.find(op_name) != std::string::npos) {
      while (std::getline(trace_file, line)) {
        if (line.find(metadata_name) != std::string::npos) {
          if (line.find(metadata_val) != std::string::npos ||
              !metadata_val.size()) {
            /* if found the right metadata_val OR if expected
             * metadata value is an empty string then ignore the matadata_val */
            return true;
          }
        }
      }
    }
  }
  return false;
}
} // namespace

TEST(MobileProfiler, ModuleHierarchy) {
  auto testModelFile = build::getResourcePath("caffe2/test/cpp/lite_interpreter_runtime/to_be_profiled_module.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        false, // profile memory
        true, // record callstack
        false, // record flops
        true); // record module hierarchy
    bc.forward(inputs);
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  const std::string metadata_name("Module Hierarchy");
  ASSERT_TRUE(checkMetaData(
      "aten::sub",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.aten::sub",
      trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  ASSERT_TRUE(checkMetaData(
      "aten::mul",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method.aten::mul",
      trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.aten::add",
      trace_file));
  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.SELF(C)::call_b.B0(B)::forward.aten::add",
      trace_file));
  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(C)::<unknown>.aten::add", trace_file));
}

TEST(MobileProfiler, Backend) {
  auto testModelFile = build::getResourcePath("caffe2/test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_backend.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        false, // profile memory
        true, // record callstack
        false, // record flops
        true); // record module hierarchy
    bc.forward(inputs);
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  std::string metadata_name("Module Hierarchy");
  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(m)::<unknown>.aten::add", trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  metadata_name = "Backend";
  ASSERT_TRUE(
      checkMetaData("aten::add", metadata_name, "test_backend", trace_file));
}

TEST(MobileProfiler, BackendMemoryEvents) {
  auto testModelFile = build::getResourcePath("caffe2/test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_backend_memory.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    mobile::KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        true, // profile memory
        true, // record callstack
        false, // record flops
        true); // record module hierarchy
    bc.forward(inputs);
  }
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  std::string metadata_name("Bytes");
  ASSERT_TRUE(checkMetaData("[memory]", metadata_name, "16384", trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  metadata_name = "Total Reserved";
  ASSERT_TRUE(checkMetaData("[memory]", metadata_name, "49152", trace_file));
}

TEST(MobileProfiler, ProfilerEvent) {
  auto testModelFile = build::getResourcePath("caffe2/test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_profiler_event.trace");

  std::vector<std::string> events(
      torch::profiler::ProfilerPerfEvents.begin(),
      torch::profiler::ProfilerPerfEvents.end());

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    // Bail if something goes wrong here
    try {
      KinetoEdgeCPUProfiler profiler(
          bc,
          trace_file_name,
          false, // record input_shapes
          false, // profile memory
          true, // record callstack
          false, // record flops
          true, // record module hierarchy
          events); // performance events
      bc.forward(inputs);
    } catch (...) {
      return;
    }
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());

  for (auto& event : events) {
    trace_file.seekg(0, std::ios_base::beg);
    /*
     * Just checking if the event entry exists in the chrometrace.
     * Checking the value in a hardware independent matter is tricky.
     */
    ASSERT_TRUE(checkMetaData("aten::__getitem__", event, "", trace_file));
  }
}

} // namespace mobile
} // namespace jit
} // namespace torch
#endif
