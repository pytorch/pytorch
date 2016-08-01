#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

class OneHotOp : public Operator<CPUContext> {
 public:
  OneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& indices = Input(0);
    auto& index_size_tensor = Input(1);
    CAFFE_ENFORCE(indices.ndim() == 1);
    CAFFE_ENFORCE(index_size_tensor.size() == 1);
    auto batch_size = indices.size();
    auto index_size = *index_size_tensor.data<int64_t>();

    auto* indices_ptr = indices.data<int64_t>();
    auto* one_hots = Output(0);
    one_hots->Resize(std::vector<TIndex>{batch_size, index_size});
    if (one_hots->size() == 0) {
      return true;
    }
    auto* one_hots_ptr = one_hots->mutable_data<float>();
    memset(one_hots_ptr, 0, one_hots->nbytes());
    for (int i = 0; i < batch_size; ++i) {
      auto label_idx = indices_ptr[i];
      DCHECK((0 <= label_idx) && (label_idx < index_size));
      one_hots_ptr[label_idx] = 1.0;
      one_hots_ptr += index_size;
    }
    return true;
  }
};

class SegmentOneHotOp : public Operator<CPUContext> {
 public:
  SegmentOneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& lengths = Input(0);
    auto& indices = Input(1);
    auto& index_size_tensor = Input(2);
    CAFFE_ENFORCE(lengths.ndim() == 1);
    CAFFE_ENFORCE(indices.ndim() == 1);
    CAFFE_ENFORCE(index_size_tensor.size() == 1);
    auto batch_size = lengths.size();
    auto index_size = *index_size_tensor.data<int64_t>();
    CAFFE_ENFORCE(index_size > 0);

    auto* lengths_ptr = lengths.data<int32_t>();
    auto* indices_ptr = indices.data<int64_t>();
    auto* one_hots = Output(0);
    one_hots->Resize(std::vector<TIndex>{batch_size, index_size});
    auto* one_hots_ptr = one_hots->mutable_data<float>();
    if (one_hots->size() == 0) {
      return true;
    }
    memset(one_hots_ptr, 0, one_hots->nbytes());
    int el_idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < lengths_ptr[i]; ++j) {
        DCHECK(el_idx < indices.size());
        auto label_idx = indices_ptr[el_idx++];
        DCHECK((0 <= label_idx) && (label_idx < index_size));
        one_hots_ptr[label_idx] = 1.0;
      }
      one_hots_ptr += index_size;
    }
    return true;
  }
};

REGISTER_CPU_OPERATOR(OneHot, OneHotOp);
REGISTER_CPU_OPERATOR(SegmentOneHot, SegmentOneHotOp);

OPERATOR_SCHEMA(OneHot)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a sequence of indices, one for each example in a batch, returns a matrix
where each inner dimension has the size of the index and has 1.0 in the index
active in the given example, and 0.0 everywhere else.
)DOC")
    .Input(0, "indices", "The active index for each example in the batch.")
    .Input(1, "index_size_tensor", "Scalar with the size of the index.")
    .Output(0, "one_hots", "Matrix of size len(indices) x index_size");

OPERATOR_SCHEMA(SegmentOneHot)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a sequence of indices, segmented by the lengths tensor, returns a matrix
that has the elements in each sequence set to 1.0, and 0.0 everywhere else.
)DOC")
    .Input(0, "lengths", "Size of each segment.")
    .Input(1, "indices", "Active indices, of size sum(lengths)")
    .Input(2, "index_size_tensor", "Size of the index")
    .Output(0, "one_hots", "Matrix of size len(lengths) x index_size");

NO_GRADIENT(OneHot);
NO_GRADIENT(SegmentOneHot);
}
}
