/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include "caffe2/core/net_async_tracing.h"

namespace caffe2 {

namespace tracing {

void testExtractShardId(const string& name, int expectedId) {
  EXPECT_EQ(extractShardId(name), expectedId);
}

TEST(NetAsyncTracingTest, ExtractShardId) {
  testExtractShardId("ABCDEFshard:1705!!A", 1705);
  // Should use the last one
  testExtractShardId("ABCDEFshard:4324!!Ashard:01220b", 1220);
  // Nothing to extract
  testExtractShardId("ABCDEFsha:222", -1);
  // Regular cases
  testExtractShardId("FC:shard:0", 0);
  testExtractShardId("FC:shard:10", 10);
  testExtractShardId("FC:shard:15", 15);
}

TEST(NetAsyncTracingTest, EveryKIteration) {
  const auto spec = R"DOC(
      name: "example"
      type: "async_scheduling"
      arg {
        name: "enable_tracing"
        i: 1
      }
      arg {
        name: "tracing_mode"
        s: "EVERY_K_ITERATIONS"
      }
      arg {
        name: "tracing_filepath"
        s: "/tmp"
      }
      arg {
        name: "trace_every_nth_batch"
        i: 1
      }
      arg {
        name: "dump_every_nth_batch"
        i: 1
      }
      op {
        output: "out"
        type: "UniformFill"
      }
)DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  Workspace ws;
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  net->Run();
}

TEST(NetAsyncTracingTest, GlobalTimeSlice) {
  const auto spec = R"DOC(
      name: "example"
      type: "async_scheduling"
      arg {
        name: "enable_tracing"
        i: 1
      }
      arg {
        name: "tracing_filepath"
        s: "/tmp"
      }
      arg {
        name: "trace_for_n_ms"
        i: 1
      }
      arg {
        name: "trace_every_n_ms"
        i: 1
      }
      op {
        output: "out"
        type: "UniformFill"
      }
)DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  Workspace ws;
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  net->Run();
}

} // namespace tracing

} // namespace caffe2
