#ifndef CAFFE2_OPERATORS_H_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_H_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/hsm.pb.h"

namespace caffe2 {

template <typename T, class Context>
class HSoftmaxOp final : public Operator<Context> {
 public:
   USE_OPERATOR_CONTEXT_FUNCTIONS;
   HSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
       : Operator<Context>(operator_def, ws) {
         hierarchy_.ParseFromString(OperatorBase::GetSingleArgument<string>
           ("hierarchy", ""));
       }
  bool RunOnDevice() override;

 private:
  HierarchyProto hierarchy_;
  Tensor<Context> scale_;
  Tensor<Context> sum_multiplier_;
  Tensor<Context> bias_multiplier_;
  DISABLE_COPY_AND_ASSIGN(HSoftmaxOp);
  float RunForwardSingle(const float* X, const float* W, const float* b,
    int target, float* output, const float* bias_multiplier, int w_length,
    int K, int& output_offset);
  static constexpr T kLOG_THRESHOLD() { return 1e-20; }
  //TODO(Deepak): Make search more efficient, maybe?
  static std::unordered_map<int, PathProto> getHierarchyForLabels(int M,
    const int* labels, const HierarchyProto& hierarchy) {
    std::unordered_map<int, PathProto> hierarchy_map;
    std::set<int> label_set = std::set<int>(labels, labels + M);
    for (const PathProto& path : hierarchy.paths()) {
      if (label_set.count(path.word_id()) > 0) {
        hierarchy_map.emplace(path.word_id(), path);
      }
    }
    return hierarchy_map;
  }
  int getIntermediateOutputSize(const int* labels, int M,
    std::unordered_map<int, PathProto>& hierarchy) {
      int size = 0;
      for (int label = 0; label < M; ++label) {
        int word_id = labels[label];
        const auto& path = hierarchy[word_id];
        size += std::accumulate(path.path_nodes().begin(),
          path.path_nodes().end(), 0,
          //Output of FC + Output of Softmax
          [](int size, PathNodeProto node) { return size + 2*node.length();});
      }
      return size;
    }
};

template <typename T, class Context>
class HSoftmaxGradientOp final : public Operator<Context> {
 public:
   USE_OPERATOR_CONTEXT_FUNCTIONS;
   HSoftmaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
       : Operator<Context>(operator_def, ws) {
         hierarchy_.ParseFromString(OperatorBase::GetSingleArgument<string>
           ("hierarchy", ""));
       }
  bool RunOnDevice() override;

 private:
  HierarchyProto hierarchy_;
  Tensor<Context> scale_;
  Tensor<Context> sum_multiplier_;
  Tensor<Context> bias_multiplier_;
  DISABLE_COPY_AND_ASSIGN(HSoftmaxGradientOp);
  void RunBackwardSingle(const float* X, const float* dY, const float* W,
    int target, const float* int_output, float* dX, float* dW,
    float* db, float* dOutput, int dim_in, int w_length, int& output_offset);
  static constexpr T kLOG_THRESHOLD() { return 1e-20; }
  //TODO(Deepak): Make search more efficient, maybe?
  static std::unordered_map<int, PathProto> getHierarchyForLabels(int M,
    const int* labels, const HierarchyProto& hierarchy) {
    std::unordered_map<int, PathProto> hierarchy_map;
    std::set<int> label_set = std::set<int>(labels, labels + M);
    for (const PathProto& path : hierarchy.paths()) {
      if (label_set.count(path.word_id()) > 0) {
        hierarchy_map.emplace(path.word_id(), path);
      }
    }
    return hierarchy_map;
  }
  int getIntermediateOutputSize(const int* labels, int M,
    std::unordered_map<int, PathProto>& hierarchy) {
      int size = 0;
      for (int label = 0; label < M; ++label) {
        int word_id = labels[label];
        const auto& path = hierarchy[word_id];
        size += std::accumulate(path.path_nodes().begin(),
          path.path_nodes().end(), 0,
          //Output of FC + Output of Softmax
          [](int size, PathNodeProto node) { return size + 2*node.length();});
      }
      return size;
    }
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_SOFTMAX_OP_H_
