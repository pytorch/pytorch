#ifndef CAFFE2_OPERATORS_DO_OP_H_
#define CAFFE2_OPERATORS_DO_OP_H_

#include <string>
#include <unordered_map>
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
      : Operator<Context>(operator_def, ws) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<NetDef>("net"),
        "net must be specified in Do operator");
    net_def_ = this->template GetSingleArgument<NetDef>("net", NetDef());

    const auto& input_names = getInputBlobNames(operator_def);
    const auto& output_names = getOutputBlobNames(operator_def);
    std::vector<std::string> outer_blob_names;
    outer_blob_names.reserve(input_names.size() + output_names.size());
    outer_blob_names.insert(
        outer_blob_names.end(), input_names.begin(), input_names.end());
    outer_blob_names.insert(
        outer_blob_names.end(), output_names.begin(), output_names.end());

    const auto& inner_blobs =
        this->template GetRepeatedArgument<std::string>("inner_blobs");
    // [0..input_names.size()-1] indices encode input blobs;
    // [input_names.size()..output_names.size()+input_names.size()-1] -
    //    encode output blobs
    const auto& outer_blobs =
        this->template GetRepeatedArgument<int>("outer_blobs_idx");
    CAFFE_ENFORCE_EQ(
        inner_blobs.size(),
        outer_blobs.size(),
        "Invalid blob bindings: different inner/outer blobs lengths");
    std::unordered_map<std::string, std::string> blob_bindings;
    for (size_t blob_idx = 0; blob_idx < inner_blobs.size(); ++blob_idx) {
      CAFFE_ENFORCE(
          !blob_bindings.count(inner_blobs[blob_idx]),
          "Invalid blob bindings: redefinition of inner blob " +
              inner_blobs[blob_idx]);
      CAFFE_ENFORCE(
          outer_blobs[blob_idx] >= 0 &&
              outer_blobs[blob_idx] < outer_blob_names.size(),
          "Invalid blob bindings: outer blob index (" +
              caffe2::to_string(outer_blobs[blob_idx]) + ", inner name: " +
              inner_blobs[blob_idx] + ") is out of bounds [0, " +
              caffe2::to_string(outer_blob_names.size() - 1) + "]");
      blob_bindings[inner_blobs[blob_idx]] =
          outer_blob_names[outer_blobs[blob_idx]];
    }

    net_workspace_.reset(new Workspace(ws, blob_bindings));
    CAFFE_ENFORCE(net_workspace_, "Failed to initialize subnet workspace");
    net_ = net_workspace_->CreateNet(net_def_, true);
    CAFFE_ENFORCE(net_, "Failed to initialize subnet");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  static std::vector<std::string> getInputBlobNames(
      const OperatorDef& operator_def) {
    std::vector<std::string> names;
    names.reserve(operator_def.input_size());
    for (auto idx = 0; idx < operator_def.input_size(); ++idx) {
      names.push_back(operator_def.input(idx));
    }
    return names;
  }

  static std::vector<std::string> getOutputBlobNames(
      const OperatorDef& operator_def) {
    std::vector<std::string> names;
    names.reserve(operator_def.output_size());
    for (auto idx = 0; idx < operator_def.output_size(); ++idx) {
      names.push_back(operator_def.output(idx));
    }
    return names;
  }

  NetDef net_def_;
  NetBase* net_;
  std::unique_ptr<Workspace> net_workspace_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DO_OP_H_
