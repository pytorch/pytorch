#ifndef CAFFE2_OPERATORS_DO_OP_H_
#define CAFFE2_OPERATORS_DO_OP_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/create_scope_op.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

template <class Context>
class DoOp final : public Operator<Context> {
 public:
  explicit DoOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), parent_ws_(ws) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<NetDef>("net"),
        "net must be specified in Do operator");
    net_def_ = this->template GetSingleArgument<NetDef>("net", NetDef());
    is_gradient_op_ = operator_def.is_gradient_op();
    copy_external_blobs_ =
        this->template GetSingleArgument<bool>("copy_external_blobs", false);
    reuse_workspace_ =
        this->template GetSingleArgument<bool>("reuse_workspace", false);
    CAFFE_ENFORCE(
        !(is_gradient_op_ && reuse_workspace_),
        "Gradient Do op requires use of stacked workspaces");
    CAFFE_ENFORCE(
        !(copy_external_blobs_ && reuse_workspace_),
        "Reuse workspace and copy external blobs simultaneously in Do op");

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
              c10::to_string(outer_blobs_idx[blob_idx]) + ", inner name: " +
              inner_blobs[blob_idx] + ") is out of bounds [0, " +
              c10::to_string(outer_blob_names.size() - 1) + "]");
      const auto& outer_name = outer_blob_names[outer_blobs_idx[blob_idx]];
      CAFFE_ENFORCE(
          !used_outer_names.count(outer_name),
          "Reusage of outer name: " + outer_name);
      used_outer_names.insert(outer_name);
      blob_bindings_[inner_blobs[blob_idx]] = outer_name;
      forwarded_inner_blobs_.insert(inner_blobs[blob_idx]);
    }
    std::unordered_set<std::string> all_outer_names(
        outer_blob_names.begin(), outer_blob_names.end());
    CAFFE_ENFORCE_EQ(
        used_outer_names.size(),
        all_outer_names.size(),
        "Not all outer names are used in blob bindings");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto* ws_stack =
        this->template Output<detail::WorkspaceStack>(OutputSize() - 1);
    std::shared_ptr<Workspace> net_workspace;
    if (is_gradient_op_) {
      net_workspace =
          ws_stack->popGradientWorkspace(parent_ws_, blob_bindings_);
    } else {
      if (reuse_workspace_ && !ws_stack->empty()) {
        net_workspace =
            ws_stack->reuseLastForwardWorkspace(parent_ws_, blob_bindings_);
      } else {
        net_workspace =
            ws_stack->pushForwardWorkspace(parent_ws_, blob_bindings_);
      }
    }
    CAFFE_ENFORCE(net_workspace, "Failed to initialize Do op workspace");

    // TODO(iliacher): figure how to reuse existing net with a new workspace
    auto* net = net_workspace->GetNet(net_def_.name());
    if (!net) {
      net = net_workspace->CreateNet(net_def_, true);
    }
    CAFFE_ENFORCE(net, "Failed to initialize subnet");
    auto success = net->Run();
    if (!is_gradient_op_ && copy_external_blobs_) {
      net_workspace->template CopyForwardedTensors<Context>(
          forwarded_inner_blobs_);
    }
    return success;
  }

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
  std::unordered_set<std::string> forwarded_inner_blobs_;
  bool is_gradient_op_;
  bool copy_external_blobs_;
  bool reuse_workspace_;
  NetDef net_def_;
  Workspace* parent_ws_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DO_OP_H_
