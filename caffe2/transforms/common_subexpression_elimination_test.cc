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
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/transforms/common_subexpression_elimination.h"

namespace caffe2 {

namespace {

using transform::Graph;

/**
 *            /--->(FC)-->(Relu)
 *  Before: (FC)-->(FC)-->(Relu)
 *            \--->(FC)-->(Relu)
 *
 *                    /-->(Relu)
 *  After : (FC)-->(FC)-->(Relu)
 *                    \-->(Relu)
 *
 */
TEST(CommonSubexpressionEliminationTest, TestSimple) {
  NetDef netdef;
  OperatorDef* op;

  // This operator simply reads input and outputs it.
  op = AddOp(&netdef, "FC", {"in", "w", "b"}, {"in1"});
  op = AddOp(&netdef, "FC", {"in1", "w", "b"}, {"mid1"});
  op = AddOp(&netdef, "FC", {"in1", "w", "b"}, {"mid2"});
  op = AddOp(&netdef, "FC", {"in1", "w", "b"}, {"mid3"});
  op = AddOp(&netdef, "Relu", {"mid1"}, {"out1"});
  op = AddOp(&netdef, "Relu", {"mid2"}, {"out2"});
  op = AddOp(&netdef, "Relu", {"mid3"}, {"out3"});

  auto t = TransformRegistry()->Create("CommonSubexpressionElimination");
  CHECK(t);
  NetDef transformed_netdef = t->ApplyTo(netdef);

  EXPECT_EQ(t->PatternMatch(Graph(netdef)).size(), 1); // one match
  EXPECT_EQ(t->PatternMatch(Graph(netdef)).at(0).size(), 3); // 3 ops matched
  EXPECT_EQ(transformed_netdef.op_size(), 5);
  EXPECT_EQ(transformed_netdef.op(1).output_size(), 1);
  EXPECT_EQ(transformed_netdef.op(2).input_size(), 1);
  EXPECT_EQ(transformed_netdef.op(3).input_size(), 1);
  EXPECT_EQ(transformed_netdef.op(4).input_size(), 1);

  // make sure op 1 writes to the blob read by 2, 3, and 4.
  EXPECT_EQ(
      transformed_netdef.op(1).output(0), transformed_netdef.op(2).input(0));
  EXPECT_EQ(
      transformed_netdef.op(1).output(0), transformed_netdef.op(3).input(0));
  EXPECT_EQ(
      transformed_netdef.op(1).output(0), transformed_netdef.op(4).input(0));
}

/**
 * Almost the same as the one above, but it has to be able to merge from
 * external input as well.
 *
 *            ->(FC)-->(Relu)
 *  Before:   ->(FC)-->(Relu)
 *            ->(FC)-->(Relu)
 *
 *                 /-->(Relu)
 *  After :   ->(FC)-->(Relu)
 *                 \-->(Relu)
 *
 */
TEST(CommonSubexpressionEliminationTest, TestFromExternal) {
  NetDef netdef;
  OperatorDef* op;

  // This operator simply reads input and outputs it.
  op = AddOp(&netdef, "FC", {"in", "w", "b"}, {"mid1"});
  op = AddOp(&netdef, "FC", {"in", "w", "b"}, {"mid2"});
  op = AddOp(&netdef, "FC", {"in", "w", "b"}, {"mid3"});
  op = AddOp(&netdef, "Relu", {"mid1"}, {"out1"});
  op = AddOp(&netdef, "Relu", {"mid2"}, {"out2"});
  op = AddOp(&netdef, "Relu", {"mid3"}, {"out3"});

  auto t = TransformRegistry()->Create("CommonSubexpressionElimination");
  CHECK(t);
  NetDef transformed_netdef = t->ApplyTo(netdef);

  EXPECT_EQ(t->PatternMatch(Graph(netdef)).size(), 1); // one match
  EXPECT_EQ(t->PatternMatch(Graph(netdef)).at(0).size(), 3); // 3 ops matched
  EXPECT_EQ(transformed_netdef.op_size(), 4);
  EXPECT_EQ(transformed_netdef.op(0).output_size(), 1);
  EXPECT_EQ(transformed_netdef.op(1).input_size(), 1);
  EXPECT_EQ(transformed_netdef.op(2).input_size(), 1);
  EXPECT_EQ(transformed_netdef.op(3).input_size(), 1);

  EXPECT_EQ(
      transformed_netdef.op(0).output(0), transformed_netdef.op(1).input(0));
  EXPECT_EQ(
      transformed_netdef.op(0).output(0), transformed_netdef.op(2).input(0));
  EXPECT_EQ(
      transformed_netdef.op(0).output(0), transformed_netdef.op(3).input(0));
}

} // namespace

} // namespace Caffe2
