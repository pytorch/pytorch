/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include "include/GenericTraceActivity.h"
#include "include/TraceSpan.h"
#include "src/output_json.h"
#include "test/TestUtils.h"

using namespace KINETO_NAMESPACE;
using namespace libkineto;

namespace {

// Exposes the protected finalizeTrace(endTime) overload so the test can flush a
// trace without constructing a Config / ActivityBuffers.
class TestableChromeTraceLogger : public ChromeTraceLogger {
 public:
  explicit TestableChromeTraceLogger(const std::string& file)
      : ChromeTraceLogger(file) {}
  using ChromeTraceLogger::finalizeTrace;
};

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  return std::string(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Write a trace containing a single CPU-op event with the given name, parse it
// back, and assert the trace is valid JSON and the name round-trips exactly.
void expectEventNameRoundTrips(const std::string& eventName) {
  const auto traceFile =
      libkineto::test::createTempTraceFile("OutputJsonTest.", ".json");

  TraceSpan span(0, 0, "test_span");
  GenericTraceActivity act(span, ActivityType::CPU_OP, eventName);
  act.startTime = 100;
  act.endTime = 200;
  act.device = 0;
  act.resource = 0;

  TestableChromeTraceLogger logger(traceFile.path());
  logger.handleTraceStart({}, "");
  logger.handleGenericActivity(act);
  logger.finalizeTrace(/*endTime=*/300);

  nlohmann::json data;
  ASSERT_NO_THROW(data = nlohmann::json::parse(readFile(traceFile.path())));

  bool found = false;
  for (const auto& event : data["traceEvents"]) {
    if (event.contains("ph") && event["ph"] == "X" && event.contains("name") &&
        event["name"].get<std::string>() == eventName) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "event name did not round-trip: " << eventName;
}

// Write a trace with one CONCURRENT_KERNEL GPU event linked to a
// record_param_comms CPU op carrying collective metadata, flush it, and parse
// the result. `quoted` selects whether string fields use the legacy RawJson
// path with surrounding double quotes or the typed path with bare values.
// Kineto must emit identical valid JSON either way.
nlohmann::json writeCollectiveTrace(bool quoted) {
  const auto traceFile =
      libkineto::test::createTempTraceFile("OutputJsonTest.", ".json");

  TraceSpan span(0, 0, "test_span");
  GenericTraceActivity recordOp(
      span, ActivityType::CPU_OP, "record_param_comms");
  auto addStringMetadata = [&](std::string_view key, const std::string& value) {
    if (quoted) {
      recordOp.addMetadata(std::string{key}, "\"" + value + "\"");
    } else {
      recordOp.addMetadata(MetadataField<std::string>{key}, value);
    }
  };
  addStringMetadata("Collective name", "allreduce");
  addStringMetadata("dtype", "Float");
  addStringMetadata("Process Group Name", "0");
  addStringMetadata("Process Group Description", "default_pg");
  addStringMetadata("In split size", "[1, 2]");
  addStringMetadata("Out split size", "[3, 4]");
  addStringMetadata("Process Group Ranks", "[0, 1, 2, 3, 4, 5, 6, 7]");
  addStringMetadata("Input Tensors start", "[[4096, 8192]]");
  addStringMetadata("Output Tensors start", "[[12288]]");
  recordOp.addMetadata(MetadataField<int64_t>{"In msg nelems"}, int64_t{1024});
  recordOp.addMetadata(MetadataField<int64_t>{"Out msg nelems"}, int64_t{512});
  recordOp.addMetadata(MetadataField<int64_t>{"Group size"}, int64_t{8});
  recordOp.addMetadata(MetadataField<int64_t>{"Rank"}, int64_t{0});
  recordOp.addMetadata(MetadataField<int64_t>{"Src Rank"}, int64_t{1});
  recordOp.addMetadata(MetadataField<int64_t>{"Dst Rank"}, int64_t{2});
  recordOp.addMetadata(MetadataField<int64_t>{"Seq"}, int64_t{3});
  recordOp.addMetadata(MetadataField<uint64_t>{"Comms Id"}, uint64_t{4});

  GenericTraceActivity kernel(
      span, ActivityType::CONCURRENT_KERNEL, "nccl:all_reduce");
  kernel.startTime = 100;
  kernel.endTime = 200;
  kernel.device = 0;
  kernel.resource = 0;
  kernel.linked = &recordOp;

  TestableChromeTraceLogger logger(traceFile.path());
  logger.handleTraceStart({}, "");
  logger.handleActivity(kernel);
  logger.finalizeTrace(/*endTime=*/300);

  return nlohmann::json::parse(readFile(traceFile.path()));
}

// Return the "args" object of the single collective GPU-kernel event.
nlohmann::json collectiveArgs(const nlohmann::json& trace) {
  for (const auto& event : trace["traceEvents"]) {
    if (event.contains("name") && event["name"] == "nccl:all_reduce" &&
        event.contains("args")) {
      return event["args"];
    }
  }
  return nlohmann::json::object();
}

} // namespace

// Reproduces pytorch/pytorch#146900: with with_stack=True an event name can
// contain `"` (e.g. dynamo co_filenames like {"device": "cpu"}). The writer
// must JSON-escape those quotes so the trace stays valid JSON.
TEST(OutputJsonTest, EventNameWithQuotesProducesValidJson) {
  expectEventNameRoundTrips(
      R"({"device": "cpu", "dtype": "float32"}(12): run)");
}

TEST(OutputJsonTest, PlainEventNameIsUnchanged) {
  expectEventNameRoundTrips("aten::addmm");
}

// Collective strings arrive quoted from the legacy RawJson path and unquoted
// from the typed path. Kineto must strip and re-quote them so the GPU-kernel
// args and distributedInfo have identical string shapes.
TEST(OutputJsonTest, CollectiveMetadataToleratesQuotedAndUnquotedForms) {
  const nlohmann::json quoted = writeCollectiveTrace(/*quoted=*/true);
  const nlohmann::json unquoted = writeCollectiveTrace(/*quoted=*/false);

  const nlohmann::json expectedArgs = {
      {"Collective name", "allreduce"},
      {"In msg nelems", 1024},
      {"Out msg nelems", 512},
      {"Group size", 8},
      {"dtype", "Float"},
      {"Input Tensors start", "[[4096, 8192]]"},
      {"Output Tensors start", "[[12288]]"},
      {"In split size", "[1, 2]"},
      {"Out split size", "[3, 4]"},
      {"Process Group Name", "0"},
      {"Process Group Description", "default_pg"},
      {"Process Group Ranks", "[0, 1, 2, 3, 4, 5, 6, 7]"},
      {"Src Rank", 1},
      {"Dst Rank", 2},
      {"Seq", 3},
      {"Comms Id", 4},
  };
  const nlohmann::json expectedDistInfo = {
      {"backend", "nccl"},
      {"rank", 0},
      {"world_size", 8},
      {"pg_count", 1},
      {"pg_config",
       {{{"pg_name", "0"},
         {"pg_desc", "default_pg"},
         {"backend_config", "cuda:nccl"},
         {"pg_size", 8},
         {"ranks", "[0, 1, 2, 3, 4, 5, 6, 7]"}}}},
      {"nccl_version", "unknown"},
  };
  // Both input forms must normalize to the same collective args and
  // distributedInfo; only environment-specific fields (e.g. traceName) differ.
  EXPECT_EQ(collectiveArgs(quoted), expectedArgs);
  EXPECT_EQ(collectiveArgs(unquoted), expectedArgs);
  EXPECT_EQ(quoted["distributedInfo"], expectedDistInfo);
  EXPECT_EQ(unquoted["distributedInfo"], expectedDistInfo);
}
