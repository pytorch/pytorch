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

#include "caffe2/transforms/single_op_transform.h"

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

using transform::Graph;

bool SingleOpTransform::PatternRule(
    const Graph& g,
    const std::vector<int>& subgraph,
    int idx) {
  if (subgraph.size() == 0) {
    return MatchOperator(g.node(idx).op);
  }
  return false;
}

bool SingleOpTransform::ValidatorRule(
    const Graph& g,
    const std::vector<int>& subgraph) {
  if (subgraph.size() == 1) {
    return true;
  }
  return false;
}

bool SingleOpTransform::ReplaceRule(
    const std::vector<int>& subgraph,
    Graph* g_ptr) {
  CHECK(g_ptr);
  auto& g = *g_ptr;
  ReplaceOperator(&(g.node(subgraph[0]).op));
  return true;
}

} // namespace caffe2
