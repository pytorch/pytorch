#ifndef CAFFE2_OPERATORS_SPARSE_TO_DENSE_MASK_OP_H_
#define CAFFE2_OPERATORS_SPARSE_TO_DENSE_MASK_OP_H_

#include <algorithm>
#include <unordered_map>
#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SparseToDenseMaskBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseToDenseMaskBase(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    std::vector<int64_t> mask =
        this->template GetRepeatedArgument<int64_t>("mask");
    featuresCount_ = mask.size();

    CAFFE_ENFORCE(!mask.empty(), "mask can't be empty");
    auto biggest = *std::max_element(mask.begin(), mask.end());
    dense_.assign(std::min(kMaxDenseSize, biggest + 1), -1);
    for (int i = 0; i < mask.size(); i++) {
      int64_t id = mask[i];
      CAFFE_ENFORCE_GE(id, 0, "Only positive IDs are allowed.");
      if (id >= kMaxDenseSize) {
        CAFFE_ENFORCE(sparse_.count(id) == 0, "Duplicated id: ", id);
        sparse_[id] = i;
      } else {
        CAFFE_ENFORCE(dense_[id] == -1, "Duplicated id: ", id);
        dense_[id] = i;
      }
    }
  }

 protected:
  const int64_t kMaxDenseSize = 1024 * 128;

  std::unordered_map<int64_t, int> sparse_;
  std::vector<int> dense_;
  size_t featuresCount_;

  inline int getFeatureIdx(int64_t id) const {
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
};

template <class Context>
class SparseToDenseMaskOp : public SparseToDenseMaskBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseToDenseMaskOp(Args&&... args)
      : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...) {
    returnPresenceMask_ =
        this->template GetSingleArgument<bool>("return_presence_mask", false);
    maxSkippedRows_ = this->template GetSingleArgument<int32_t>(
        "max_skipped_indices", kMaxSkippedSparseIndices);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
    auto& sparse_values = Input(VALUES);
    CAFFE_ENFORCE_GE(sparse_values.dim(), 1);
    CAFFE_ENFORCE_EQ(sparse_indices.numel(), sparse_values.size(0));
    auto& default_value = Input(DEFAULT);
    CAFFE_ENFORCE_EQ(default_value.dim() + 1, sparse_values.dim());
    CAFFE_ENFORCE_EQ(default_value.numel(), sparse_values.size_from_dim(1));
    CAFFE_ENFORCE(sparse_values.dtype() == default_value.dtype());

    const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
    const char* sparse_values_vec =
        static_cast<const char*>(sparse_values.raw_data());
    const void* default_val = default_value.raw_data();

    int64_t block_size = default_value.numel();
    size_t block_nbytes = default_value.nbytes();

    const size_t cols = this->featuresCount_;
    int rows = -1;
    int32_t sparse_indices_length = sparse_indices.dim32(0);
    const int32_t* lengths_vec = nullptr;
    auto* output = Output(OUTPUTVALUE);
    Tensor* presence_mask = nullptr;
    if (returnPresenceMask_) {
      presence_mask = Output(PRESENCEMASK);
    }
    vector<int64_t> shape;
    if (InputSize() == 4) {
      auto& lengths = Input(LENGTHS);
      CAFFE_ENFORCE_EQ(lengths.dim(), 1);
      lengths_vec = lengths.template data<int32_t>();
      rows = lengths.dim32(0);
    }
    if (rows == -1) {
      // if the LENGTHS is not set, the output will be a vector
      rows = 1;
      lengths_vec = &sparse_indices_length;
    } else {
      shape.push_back(rows);
    }
    shape.push_back(cols);
    if (returnPresenceMask_) {
      presence_mask->Resize(shape);
    }
    shape.insert(
        shape.end(),
        default_value.sizes().begin(),
        default_value.sizes().end());
    output->Resize(shape);

    // init
    // TODO: consider unrolling CopyItems to make elemental types copy faster
    char* output_data =
        static_cast<char*>(output->raw_mutable_data(sparse_values.dtype()));
    for (int i = 0; i < cols * rows; i++) {
      context_.CopyItemsSameDevice(
          default_value.dtype(),
          block_size,
          default_val,
          output_data + i * block_nbytes);
    }
    bool* presence_mask_data = nullptr;
    if (returnPresenceMask_) {
      presence_mask_data = presence_mask->template mutable_data<bool>();
      math::Set<bool, Context>(
          rows * cols, false, presence_mask_data, &context_);
    }

    int64_t offset = 0;
    for (int r = 0; r < rows; r++) {
      bool skippedSparseIndex = false;
      for (int c = 0; c < lengths_vec[r]; c++) {
        const auto sparse_index = sparse_indices_vec[offset + c];
        if (sparse_index < 0 ||
            sparse_index >= std::numeric_limits<TInd>::max()) {
          skippedSparseIndex = true;
          LOG(WARNING) << "Skipping invalid sparse index: " << sparse_index;
          continue;
        }
        int idx = this->getFeatureIdx(sparse_index);
        if (idx != -1) {
          context_.CopyItemsSameDevice(
              sparse_values.dtype(),
              block_size,
              sparse_values_vec + (offset + c) * block_nbytes,
              output_data + (r * cols + idx) * block_nbytes);
          if (returnPresenceMask_) {
            presence_mask_data[r * cols + idx] = true;
          }
        }
      }
      skippedRows_ += skippedSparseIndex;
      CAFFE_ENFORCE_LT(
          skippedRows_,
          maxSkippedRows_,
          "Too many rows with invalid sparse indices skipped");
      offset += lengths_vec[r];
    }

    return true;
  }

 private:
  static const uint32_t kMaxSkippedSparseIndices = 50;

  bool returnPresenceMask_;
  uint32_t maxSkippedRows_ = 0;
  uint32_t skippedRows_ = 0;

  INPUT_TAGS(INDICES, VALUES, DEFAULT, LENGTHS);
  OUTPUT_TAGS(OUTPUTVALUE, PRESENCEMASK);
};

template <class Context>
class SparseToDenseMaskGradientOp : public SparseToDenseMaskBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseToDenseMaskGradientOp(Args&&... args)
      : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
    auto& gradient_output = Input(GOUTPUT);

    int64_t block_size = gradient_output.size_from_dim(1);
    size_t block_nbytes = gradient_output.itemsize() * block_size;

    const size_t cols = this->featuresCount_;
    int rows = -1;
    int iter_offset = 1;
    int32_t default_length = sparse_indices.dim32(0);
    const int32_t* lengths_vec = nullptr;
    auto* output = Output(GVALUES);
    vector<int64_t> shape;
    if (InputSize() > LENGTHS) {
      // if the LENGTHS is set, the gradient_output has dim:
      // lengths * mask.size() * feature_dim
      auto& lengths = Input(LENGTHS);
      lengths_vec = lengths.template data<int32_t>();
      rows = lengths.dim32(0);
      CAFFE_ENFORCE_EQ(lengths.dim(), 1);
      CAFFE_ENFORCE_GE(gradient_output.dim(), 2);
      CAFFE_ENFORCE_EQ(gradient_output.size(0), rows);
      CAFFE_ENFORCE_EQ(gradient_output.size(1), cols);
      block_nbytes /= gradient_output.size(1);
      block_size /= gradient_output.size(1);
      iter_offset += 1;
    }
    if (rows == -1) {
      // if the LENGTHS is not set, the gradient_output has dim:
      // mask.size() * feature_dim
      rows = 1;
      lengths_vec = &default_length;
      CAFFE_ENFORCE_GE(gradient_output.dim(), 1);
      CAFFE_ENFORCE_EQ(gradient_output.size(0), cols);
    }
    shape.push_back(default_length);
    // insert feature_dim
    shape.insert(
        shape.end(),
        gradient_output.sizes().begin() + iter_offset,
        gradient_output.sizes().end());
    output->Resize(shape);

    const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
    const char* gradient_output_vec =
        static_cast<const char*>(gradient_output.raw_data());

    char* output_data =
        static_cast<char*>(output->raw_mutable_data(gradient_output.dtype()));
    memset(output_data, 0, output->nbytes());
    math::Set<char, Context>(
        default_length * gradient_output.itemsize(), 0, output_data, &context_);

    int32_t offset = 0;
    // SparseToDenseMask is not injective; gradient_used records
    // if the gradient is used for other input value from the same row
    vector<bool> gradient_used(cols, false);
    for (int r = 0; r < rows; r++) {
      std::fill(gradient_used.begin(), gradient_used.end(), false);
      for (int c = lengths_vec[r] - 1; c >= 0; c--) {
        int idx = this->getFeatureIdx(sparse_indices_vec[offset + c]);
        if (idx != -1 && !gradient_used[idx]) {
          gradient_used[idx] = true;
          context_.CopyItemsSameDevice(
              gradient_output.dtype(),
              block_size,
              gradient_output_vec + (r * cols + idx) * block_nbytes,
              output_data + (offset + c) * block_nbytes);
        }
      }
      offset += lengths_vec[r];
    }
    return true;
  }

 private:
  INPUT_TAGS(INDICES, GOUTPUT, LENGTHS);
  OUTPUT_TAGS(GVALUES);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPARSE_TO_DENSE_MASK_OP_H_
