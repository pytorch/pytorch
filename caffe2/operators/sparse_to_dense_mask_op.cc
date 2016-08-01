#include <algorithm>
#include <unordered_map>
#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

using ValueTypes = TensorTypes<int32_t, int64_t, float, double, string, bool>;

class SparseToDenseMaskOp : public Operator<CPUContext> {
 public:
  SparseToDenseMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    std::vector<int> mask = GetRepeatedArgument<int>("mask");
    featuresCount_ = mask.size();
    auto biggest = *std::max_element(mask.begin(), mask.end());
    dense_.assign(std::min(kMaxDenseSize, biggest + 1), -1);
    for (int i = 0; i < mask.size(); i++) {
      int id = mask[i];
      CAFFE_ENFORCE(id >= 0, "Only positive IDs are allowed.");
      if (id >= kMaxDenseSize) {
        sparse_[id] = i;
      } else {
        dense_[id] = i;
      }
    }
  }

  bool RunOnDevice() override {
    const TypeMeta& meta = Input(INDICES).meta();
    if (meta.Match<int32_t>()) {
      return DoRunWithIndexType<int32_t>();
    } else if (meta.Match<int64_t>()) {
      return DoRunWithIndexType<int64_t>();
    } else {
      CAFFE_THROW("Unsupported type of tensor: ", meta.name());
      return false;
    }
  }

  template <typename TInd>
  bool DoRunWithIndexType() {
    if (InputSize() < 4) {
      return DoRunWithLengthType<TInd, int32_t>();
    } else {
      const TypeMeta& meta = Input(LENGTHS).meta();
      if (meta.Match<int32_t>()) {
        return DoRunWithLengthType<TInd, int32_t>();
      } else if (meta.Match<int64_t>()) {
        return DoRunWithLengthType<TInd, int64_t>();
      } else {
        CAFFE_THROW("Unsupported type of tensor: ", meta.name());
        return false;
      }
    }
  }

  template <typename TInd, typename TLen>
  bool DoRunWithLengthType() {
    return DispatchHelper<ValueTypes, TInd, TLen>::call(this, Input(VALUES));
  }

  template <typename TInd, typename TLen, typename TVal>
  bool DoRunWithType() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE(sparse_indices.ndim() == 1);
    auto& sparse_values = Input(VALUES);
    CAFFE_ENFORCE(sparse_values.ndim() == 1);
    CAFFE_ENFORCE(sparse_indices.size() == sparse_values.size());
    auto& default_value = Input(DEFAULT);
    CAFFE_ENFORCE(default_value.size() == 1);

    const TInd* sparse_indices_vec = sparse_indices.data<TInd>();
    const TVal* sparse_values_vec = sparse_values.template data<TVal>();
    const TVal* default_val = default_value.template data<TVal>();

    int cols = featuresCount_;
    int rows = 0;
    TLen default_length = sparse_indices.dim32(0);
    const TLen* lengths_vec = nullptr;
    auto* output = Output(0);
    if (InputSize() == 4) {
      auto& lengths = Input(LENGTHS);
      CAFFE_ENFORCE(lengths.ndim() == 1);
      lengths_vec = lengths.data<TLen>();
      rows = lengths.dim32(0);
      output->Resize(rows, cols);
    }
    if (rows == 0) {
      // if the LENGTHS is not set or it is empty, the output will be a vector
      rows = 1;
      lengths_vec = &default_length;
      output->Resize(cols);
    }

    // init
    TVal* output_data = output->template mutable_data<TVal>();
    for (int i = 0; i < cols * rows; i++) {
      output_data[i] = default_val[0];
    }

    TLen offset = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < lengths_vec[r]; c++) {
        int idx = getFeatureIdx(sparse_indices_vec[offset + c]);
        if (idx != -1) {
          output_data[r * cols + idx] = sparse_values_vec[offset + c];
        }
      }
      offset += lengths_vec[r];
    }

    return true;
  }

 private:
  const int kMaxDenseSize = 1024 * 128;

  std::unordered_map<int, int> sparse_;
  std::vector<int> dense_;
  int featuresCount_;

  inline int getFeatureIdx(int id) const {
    if (id >= kMaxDenseSize) {
      const auto& iter = sparse_.find(id);
      if (iter == sparse_.end()) {
        return -1;
      } else {
        return iter->second;
      }
    } else {
      return (id >= dense_.size()) ? -1 : dense_[id];
    }
  }

  INPUT_TAGS(INDICES, VALUES, DEFAULT, LENGTHS);
};

namespace {
REGISTER_CPU_OPERATOR(SparseToDenseMask, SparseToDenseMaskOp);

OPERATOR_SCHEMA(SparseToDenseMask)
    .NumInputs(3, 4)
    .NumOutputs(1)
    .SetDoc("Convert sparse representations to dense with given indices.")
    .Output(0, "output", "1-D or 2-D dense tensor.");

NO_GRADIENT(SparseToDenseMask);
} // namespace
} // namespace caffe2
