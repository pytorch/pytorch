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

#ifndef CAFFE2_OPERATORS_DO_OP_H_
#define CAFFE2_OPERATORS_DO_OP_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

template <class Context>
class DoOp final : public Operator<Context> {
 public:
  DoOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), parent_ws_(ws) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<NetDef>("net"),
        "net must be specified in Do operator");
    net_def_ = this->template GetSingleArgument<NetDef>("net", NetDef());
    is_gradient_op_ = operator_def.is_gradient_op();
    reuse_workspace_ =
        this->template GetSingleArgument<bool>("reuse_workspace", false);
    CAFFE_ENFORCE(
        !is_gradient_op_ || !reuse_workspace_,
        "Gradient Do op requires use of stacked workspaces");

    const auto& inner_blobs =
        this->template GetRepeatedArgument<std::string>("inner_blobs");
    const auto& outer_blobs_idx =
        this->template GetRepeatedArgument<int>("outer_blobs_idx");
    CAFFE_ENFORCE_EQ(
        inner_blobs.size(),
        outer_blobs_idx.size(),
        "Invalid blob bindings: different inner/outer blobs lengths");

    const auto& outer_blob_names = checkAndGetOuterNames(operator_def);
    std::unordered_set<std::string> used_outer_names;
    for (size_t blob_idx = 0; blob_idx < inner_blobs.size(); ++blob_idx) {
      CAFFE_ENFORCE(
          !blob_bindings_.count(inner_blobs[blob_idx]),
          "Invalid blob bindings: redefinition of inner blob " +
              inner_blobs[blob_idx]);
      CAFFE_ENFORCE(
          outer_blobs_idx[blob_idx] >= 0 &&
              outer_blobs_idx[blob_idx] < outer_blob_names.size(),
          "Invalid blob bindings: outer blob index (" +
              caffe2::to_string(outer_blobs_idx[blob_idx]) + ", inner name: " +
              inner_blobs[blob_idx] + ") is out of bounds [0, " +
              caffe2::to_string(outer_blob_names.size() - 1) + "]");
      const auto& outer_name = outer_blob_names[outer_blobs_idx[blob_idx]];
      CAFFE_ENFORCE(
          !used_outer_names.count(outer_name),
          "Reusage of outer name: " + outer_name);
      used_outer_names.insert(outer_name);
      blob_bindings_[inner_blobs[blob_idx]] = outer_name;
    }
    std::unordered_set<std::string> all_outer_names(
        outer_blob_names.begin(), outer_blob_names.end());
    CAFFE_ENFORCE_EQ(
        used_outer_names.size(),
        all_outer_names.size(),
        "Not all outer names are used in blob bindings");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  // returns vector of input blob names followed by output blob names in
  // operator definition order; ensures that input (output) names are unique,
  // checks number of input (output) blobs
  std::vector<std::string> checkAndGetOuterNames(
      const OperatorDef& operator_def) const {
    auto input_names = getInputBlobNames(operator_def);
    CAFFE_ENFORCE(!input_names.empty(), "Expected at least one input blob");
    std::string input_ws_blob = input_names.back(); // copy
    // removing blob that holds pointer op workspace
    input_names.pop_back();

    std::unordered_set<std::string> all_input_names(
        input_names.begin(), input_names.end());
    CAFFE_ENFORCE_EQ(
        input_names.size(), all_input_names.size(), "Duplicate input blobs");

    auto output_names = getOutputBlobNames(operator_def);
    CAFFE_ENFORCE(!output_names.empty(), "Expected at least one output blob");
    const auto& output_ws_blob = output_names.back();
    CAFFE_ENFORCE_EQ(
        input_ws_blob,
        output_ws_blob,
        "Expected same input/output workspace blob");
    // remove blob that holds pointer to op workspace
    output_names.pop_back();

    std::unordered_set<std::string> all_output_names(
        output_names.begin(), output_names.end());
    CAFFE_ENFORCE_EQ(
        output_names.size(), all_output_names.size(), "Duplicate output blobs");

    std::vector<std::string> outer_blob_names;
    outer_blob_names.reserve(input_names.size() + output_names.size());
    outer_blob_names.insert(
        outer_blob_names.end(), input_names.begin(), input_names.end());
    outer_blob_names.insert(
        outer_blob_names.end(), output_names.begin(), output_names.end());
    return outer_blob_names;
  }

  std::vector<std::string> getInputBlobNames(
      const OperatorDef& operator_def) const {
    std::vector<std::string> names;
    names.reserve(operator_def.input_size());
    for (auto idx = 0; idx < operator_def.input_size(); ++idx) {
      names.push_back(operator_def.input(idx));
    }
    return names;
  }

  std::vector<std::string> getOutputBlobNames(
      const OperatorDef& operator_def) const {
    std::vector<std::string> names;
    names.reserve(operator_def.output_size());
    for (auto idx = 0; idx < operator_def.output_size(); ++idx) {
      names.push_back(operator_def.output(idx));
    }
    return names;
  }

  std::unordered_map<std::string, std::string> blob_bindings_;
  bool is_gradient_op_;
  bool reuse_workspace_;
  NetDef net_def_;
  Workspace* parent_ws_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DO_OP_H_
