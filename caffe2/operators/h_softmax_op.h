#ifndef CAFFE2_OPERATORS_H_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_H_SOFTMAX_OP_H_

#include <c10/util/Optional.h>
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/hsm.pb.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, typename Context>
class HSoftmaxOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit HSoftmaxOpBase(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    HierarchyProto hierarchy;
    CAFFE_ENFORCE(hierarchy.ParseFromString(
        this->template GetSingleArgument<string>("hierarchy", "")));
    for (const auto& path : hierarchy.paths()) {
      hierarchy_all_map_.emplace(path.word_id(), path);
    }
  }

 protected:
  std::unordered_map<int, PathProto> hierarchy_all_map_;
  c10::optional<Tensor> scale_;
  c10::optional<Tensor> sum_multiplier_;
  c10::optional<Tensor> bias_multiplier_;
  static constexpr T kLOG_THRESHOLD() {
    return 1e-20f;
  }
  static std::unordered_map<int, PathProto> getHierarchyForLabels(
      int M,
      const int* labels,
      const std::unordered_map<int, PathProto>& hierarchy_all_map) {
    std::unordered_map<int, PathProto> hierarchy_map;
    std::set<int> label_set = std::set<int>(labels, labels + M);
    for (const auto& label : label_set) {
      auto search = hierarchy_all_map.find(label);
      CAFFE_ENFORCE(search != hierarchy_all_map.end(), "incorrect label.");
      hierarchy_map.emplace(search->first, search->second);
    }
    return hierarchy_map;
  }
  int getIntermediateOutputSize(
      const int* labels,
      int M,
      std::unordered_map<int, PathProto>& hierarchy) const {
    int size = 0;
    for (int label = 0; label < M; ++label) {
      int word_id = labels[label];
      const auto& path = hierarchy[word_id];
      size += std::accumulate(
          path.path_nodes().begin(),
          path.path_nodes().end(),
          0,
          // Output of FC + Output of Softmax
          [](int sz, PathNodeProto node) { return sz + 2 * node.length(); });
    }
    return size;
  }
};

template <typename T, class Context>
class HSoftmaxOp : public HSoftmaxOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using HSoftmaxOpBase<T, Context>::HSoftmaxOpBase;

  bool RunOnDevice() override;

 protected:
  float RunForwardSingle(
      const float* X,
      const float* W,
      const float* b,
      int target,
      float* output,
      const float* bias_multiplier,
      int w_length,
      int K,
      int& output_offset);
};

template <typename T, class Context>
class HSoftmaxGradientOp final : public HSoftmaxOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using HSoftmaxOpBase<T, Context>::HSoftmaxOpBase;
  bool RunOnDevice() override;

 private:
  void RunBackwardSingle(
      const float* X,
      const float* dY,
      const float* W,
      int target,
      const float* int_output,
      float* dX,
      float* dW,
      float* db,
      float* dOutput,
      int dim_in,
      int w_length,
      int& output_offset);
};

template <typename T, class Context>
class HSoftmaxSearchOp final : public HSoftmaxOp<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit HSoftmaxSearchOp(Args&&... args)
      : HSoftmaxOp<T, Context>(std::forward<Args>(args)...),
        top_n_(this->template GetSingleArgument<int>("topN", 5)),
        beam_(this->template GetSingleArgument<float>("beam", 0.01f)) {
    CAFFE_ENFORCE(tree_.ParseFromString(
        this->template GetSingleArgument<string>("tree", "")));
  }
  bool RunOnDevice() override;

 private:
  int top_n_;
  float beam_;
  TreeProto tree_;
  bool pruning(
      const float* X,
      int sample,
      int K,
      const float* W,
      const float* b,
      const NodeProto& src_node,
      NodeProto& dst_node,
      float parent_score,
      float beam);
  bool extractNodes(
      const NodeProto& node,
      std::vector<std::pair<string, float>>& info);
};

template <typename T, class Context>
class HuffmanTreeHierarchyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit HuffmanTreeHierarchyOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        num_classes_(this->template GetSingleArgument<int>("num_classes", -1)) {
  }
  bool RunOnDevice() override;

 private:
  // Internal huffman tree data.
  struct Node {
    Node(T l, int count)
        : label(l), count(count), left_ch_index(-1), right_ch_index(-1) {}
    T label;
    int count;
    int left_ch_index;
    int right_ch_index;
  };

  struct NodeComparator {
    bool operator()(const Node& node_a, const Node& node_b) {
      return node_a.count > node_b.count;
    }
  };

  int num_classes_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SOFTMAX_OP_H_
