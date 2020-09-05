#pragma once

#include "caffe2/opt/backend_transformer_base.h"

#include <unordered_set>

namespace caffe2 {

struct TvmTransformOptions final : public BackendTransformOptions {
  explicit TvmTransformOptions() : BackendTransformOptions() {}

  //  Whether to enable profiling based jit
  bool profiling_based_jit{false};
};

class CAFFE2_API TvmTransformer final : public BackendTransformerBase {
 public:
  explicit TvmTransformer(const TvmTransformOptions& opts)
      : BackendTransformerBase(), opts_(opts) {}

  ~TvmTransformer() override {}

  // Given workspace and predict net, cluster continuous parts that can be run
  // by TVM and create one TVMJit op for each clustered subgraph.
  // \param ws c2 workspace
  // \param pred_net c2 predict net
  // \param weight_names list of the names of the constant weights
  // \param shape_hints User provided shape info, usually for primary inputs so
  // that bound shape inference can have something to start
  // \param blocklisted_ops a set of ops that we don't want to lower to TVM in
  // terms of their net positions. This is very useful for debugging but for
  // normal runs it should be empty
  void transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::vector<std::string>& weight_names,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<int>& blocklisted_ops) override;

  static const std::unordered_set<std::string>& getSupportedOps();

  static bool canConvertFullGraph(
      const caffe2::NetDef& net,
      const std::unordered_set<int>& blocklisted_ops);

 private:
  // Given TVM runnable subnets, contract them into one TVMJitOp
  NetDef buildTvmOp(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights,
      const ShapeInfoMap& shape_hints);

  // Apply transform to cluster connected TVM runnable ops into one TVMJitOp
  NetDef applyTvmTransform(
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blocklisted_ops,
      const ShapeInfoMap& shape_hints);

  // Options
  TvmTransformOptions opts_;

  // Track number of TVMJitOp we created
  int tvm_op_id_{0};

  // Model id
  std::string model_id_;
};

// Helper function to clean up a net and run tvm transform.
CAFFE2_API void tvmTransform(
    NetDef* net,
    Workspace* ws,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& weight_names,
    const ShapeInfoMap& shape_hints,
    const std::unordered_set<int>& blocklisted_ops,
    int32_t max_batch_size,
    int32_t max_seq_size,
    int32_t num_embeddings,
    int32_t embedding_size,
    int32_t tvm_min_ops,
    bool tvm_profiling_based_jit,
    bool debug);

CAFFE2_API void cleanUpPredictNet(
    NetDef* net,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& weight_names);

} // namespace caffe2
