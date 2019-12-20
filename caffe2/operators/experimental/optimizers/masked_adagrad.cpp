#include <shared_mutex>

#include "caffe2/sgd/adagrad_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

class MaskedAdagradOp final : public Operator<CPUContext> {
 public:
  MaskedAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
        decay_(this->template GetSingleArgument<float>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    int N = Input(GRAD).numel();
    const float* w = Input(PARAM).template data<float>();
    const float* g = Input(GRAD).template data<float>();
    const float* h = Input(MOMENT_1).template data<float>();
    const float* mask = Input(MASK).template data<float>();
    float* nw = Output(OUTPUT_PARAM)->template mutable_data<float>();
    float* nh = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();
    float lr = Input(LR).template data<float>()[0];
    for (auto i = 0; i < N; ++i) {
      float gi = g[i];
      float hi = decay_ * h[i] + gi * gi;
      nh[i] = mask[i] * hi;
      nw[i] = mask[i] * (w[i] + lr / (std::sqrt(hi) + epsilon_) * gi);
    }
    return true;
  }

 protected:
  float epsilon_;
  float decay_;
  INPUT_TAGS(PARAM, MOMENT_1, GRAD, LR, MASK);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <bool ROWWISE = false>
class MaskedSparseAdagradOp final : public Operator<CPUContext> {
 public:
  MaskedSparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
        block_size_(this->template GetSingleArgument<int>("block_size", 1)),
        delays_(this->template GetRepeatedArgument<int>("delays")),
        prune_ratios_(
            this->template GetRepeatedArgument<float>("prune_ratios")) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const float* lr = Input(LR).template data<float>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto n = Input(INDICES).numel();

    const SIndex* indices = Input(INDICES).template data<SIndex>();
    const float* gradIn = Input(GRAD).template data<float>();
    const float* paramIn = Input(PARAM).template data<float>();
    const float* momentIn = Input(MOMENT_1).template data<float>();
    float* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
    float* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();

    if (n == 0) {
      return true;
    }
    auto row_size = Input(GRAD).numel() / n;

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    if (ROWWISE) {
      CAFFE_ENFORCE_EQ(
          Input(PARAM).numel() / row_size,
          Input(MOMENT_1).numel(),
          "Input Param size: ",
          Input(PARAM).numel(),
          " Block size: ",
          row_size,
          " Input Moment size: ",
          Input(MOMENT_1).numel());
    } else {
      CAFFE_ENFORCE_EQ(
          Input(PARAM).numel(),
          Input(MOMENT_1).numel(),
          "Input Param size: ",
          Input(PARAM).numel(),
          " Input Moment size: ",
          Input(MOMENT_1).numel());
    }

    // input(grad) is compatible with size of indexes
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel() % n,
        0,
        "Incorrect gradient size:",
        Input(GRAD).numel(),
        " size of indexes:",
        n);

    bool mask_changed = false;
    bool write_lock_acquired = false;
    if (delays_.empty()) {
      mask_changed = Input(MASK_CHANGED).template data<bool>()[0];
    } else {
      // If delay is present, prune_rate also should be there
      CAFFE_ENFORCE(!prune_ratios_.empty());
      CAFFE_ENFORCE_EQ(delays_.size(), prune_ratios_.size());

      std::int64_t iteration = Input(ITER).template data<std::int64_t>()[0];

      auto delay_itr = std::find(delays_.begin(), delays_.end(), iteration);
      if (delay_itr != delays_.end()) {
        // If the current iteration matches with one of delays, we update mask
        float prune_ratio = prune_ratios_.at(delay_itr - delays_.begin());

        // Take an exclusive ownership of masks and weights
        lock_.lock();
        write_lock_acquired = true;
        UpdateMask_(prune_ratio);
        mask_changed = true;
      }
    }

    int num_blocks_per_row = (row_size + block_size_ - 1) / block_size_;
    int bitmask_bytes_per_row = (num_blocks_per_row + 7) / 8;

    // Get mask
    const uint8_t* bitmask = nullptr;
    if (delays_.empty()) {
      // If delays argument is empty, get from MASK input
      bitmask = Input(MASK).template data<uint8_t>();
    } else if (OutputSize() > MASK_OUT) {
      // If MASK_OUT is present, the blob stores bitmask
      if (Output(MASK_OUT)->numel()) {
        bitmask = Output(MASK_OUT)->template data<uint8_t>();
      }
    } else {
      // Otherwise, the operator keeps the mask internally.
      bitmask = bitmask_.get();
    }

    if (mask_changed && bitmask) {
      if (!write_lock_acquired) {
        lock_.lock();
        write_lock_acquired = true;
      }
      // If mask has changed, update the whole table using the new mask.
      for (int i = 0; i < Input(PARAM).dim32(0); ++i) {
        for (int j = 0; j < row_size; ++j) {
          int j_block = j / block_size_;
          int byte = j_block / 8;
          int bit = j_block % 8;
          bool mask = bitmask[i * bitmask_bytes_per_row + byte] & (1 << bit);
          if (!mask) {
            if (!ROWWISE) {
              momentOut[i * row_size + j] = 0;
            }
            paramOut[i * row_size + j] = 0;
          }
        }
      }
    }
    if (write_lock_acquired) {
      lock_.unlock();
    }

    // Acquire read lock
    lock_.lock_shared();

    // The actual SparseAdagrad update
    for (int i = 0; i < n; ++i) {
      std::size_t idx = indices[i];
      auto offsetI = i * row_size;
      auto offsetIdx = idx * row_size;

      // Enforce:
      // access within range
      // gradient access within range
      CAFFE_ENFORCE_GE(
          Input(PARAM).numel(),
          row_size + offsetIdx,
          this->debug_def().input(PARAM),
          ", out of bound,  idx:",
          idx,
          " for input i:",
          i,
          " and block size:",
          row_size,
          " max size:",
          Input(PARAM).numel());

      // TODO: performance can be optimized using intrinsics later.
      if (ROWWISE) {
        float sum = 0;
#ifdef MASKED_ADAGRAD_MASK_GRAD
        int nnz = 0;
#endif
        for (int j = 0; j < row_size; ++j) {
#ifdef MASKED_ADAGRAD_MASK_GRAD
          // Get mask
          bool mask = true;
          if (bitmask) {
            int j_block = j / block_size_;
            int byte = j_block / 8;
            int bit = j_block % 8;
            mask = bitmask[idx * bitmask_bytes_per_row + byte] & (1 << bit);
          }
          if (mask) {
            ++nnz;
            sum += gradIn[offsetI + j] * gradIn[offsetI + j];
          }
#else
          sum += gradIn[offsetI + j] * gradIn[offsetI + j];
#endif
        }
#ifdef MASKED_ADAGRAD_MASK_GRAD
        if (nnz) {
          sum /= nnz;
        }
#else
        sum /= row_size;
#endif

        float hi = momentOut[idx] = momentIn[idx] + sum;
        float float_step = lr[0] / (std::sqrt(hi) + epsilon_);

        for (int j = 0; j < row_size; ++j) {
          bool mask = true;
          if (bitmask) {
            int j_block = j / block_size_;
            int byte = j_block / 8;
            int bit = j_block % 8;
            mask = bitmask[idx * bitmask_bytes_per_row + byte] & (1 << bit);
          }
          if (mask) {
            float gi = gradIn[offsetI + j];
            paramOut[offsetIdx + j] = paramIn[offsetIdx + j] + gi * float_step;
          } else {
            paramOut[offsetIdx + j] = 0;
          }
        }
      } else {
        for (int j = 0; j < row_size; ++j) {
          // Get mask
          bool mask = true;
          if (bitmask) {
            int j_block = j / block_size_;
            int byte = j_block / 8;
            int bit = j_block % 8;
            mask = bitmask[idx * bitmask_bytes_per_row + byte] & (1 << bit);
          }

          // Actual Adagrad update
          if (mask) {
            float gi = gradIn[offsetI + j];
            float hi = momentOut[offsetIdx + j] =
                momentIn[offsetIdx + j] + gi * gi;
            paramOut[offsetIdx + j] = paramIn[offsetIdx + j] +
                lr[0] / (std::sqrt(hi) + epsilon_) * gi;
          } else {
            momentOut[offsetIdx + j] = 0;
            paramOut[offsetIdx + j] = 0;
          }
        } // !ROWWISE
      }
    }

    // Release read lock
    lock_.unlock_shared();

    return true;
  }

 private:
  void UpdateMask_(float prune_ratio) {
    auto n = Input(INDICES).numel();
    auto row_size = Input(GRAD).numel() / n;

    // Create a temp buffer to store norm square of each block
    int num_rows = Input(PARAM).numel() / row_size;
    int num_blocks_per_row = (row_size + block_size_ - 1) / block_size_;
    std::vector<float> norm_sq_buffer(num_rows * num_blocks_per_row);
    const float* param = Input(PARAM).template data<float>();
    for (int i = 0; i < num_rows; ++i) {
      for (int j_block = 0; j_block < num_blocks_per_row; ++j_block) {
        float norm_sq = 0;
        for (int j = j_block * block_size_;
             j < std::min<int>((j_block + 1) * block_size_, row_size);
             ++j) {
          norm_sq += param[i * row_size + j] * param[i * row_size + j];
        }
        norm_sq_buffer[i * num_blocks_per_row + j_block] = norm_sq;
      }
    }

    // Create a temp buffer to store partially sorted results
    int num_blocks_to_prune =
        std::floor(num_rows * num_blocks_per_row * prune_ratio);
    std::vector<float> norm_sq_partially_sorted(num_blocks_to_prune);
    std::partial_sort_copy(
        norm_sq_buffer.begin(),
        norm_sq_buffer.end(),
        norm_sq_partially_sorted.begin(),
        norm_sq_partially_sorted.end());

    // Determine threshold
    float threshold =
        norm_sq_partially_sorted.empty() ? 0 : norm_sq_partially_sorted.back();

    // Create bitmask
    uint8_t* bitmask = nullptr;
    // For each row, we start a new byte in bitmask
    int bitmask_bytes_per_row = (num_blocks_per_row + 7) / 8;

    if (OutputSize() > MASK_OUT) {
      Output(MASK_OUT)->Resize(num_rows, bitmask_bytes_per_row);
      bitmask = Output(MASK_OUT)->template mutable_data<uint8_t>();
    } else {
      if (!bitmask_) {
        bitmask_.reset(new uint8_t[num_rows * bitmask_bytes_per_row]);
      }
      bitmask = bitmask_.get();
    }

    for (int i = 0; i < num_rows; ++i) {
      for (int j_block = 0; j_block < num_blocks_per_row; ++j_block) {
        int byte = j_block / 8;
        int bit = j_block % 8;
        if (bit == 0) {
          bitmask[i * bitmask_bytes_per_row + byte] = 0;
        }
        int current_bitmask =
            norm_sq_buffer[i * num_blocks_per_row + j_block] >= threshold;
        bitmask[i * bitmask_bytes_per_row + byte] |= current_bitmask << bit;
      }
    }
  } // UpdateMask_

  float epsilon_;
  int block_size_;
  std::vector<int> delays_;
  std::vector<float> prune_ratios_;

  std::unique_ptr<uint8_t[]> bitmask_;

  // A read-write lock to prevent access to weights while mask is being updated
  // or weights are being pruned.
  // Having a single global lock is overly protective but should be good for
  // now for experimentations.
  static std::shared_mutex lock_;

  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, MASK, MASK_CHANGED, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, MASK_OUT);
};

template <bool ROWWISE>
std::shared_mutex MaskedSparseAdagradOp<ROWWISE>::lock_;

REGISTER_CPU_OPERATOR(MaskedAdagrad, MaskedAdagradOp);
OPERATOR_SCHEMA(MaskedAdagrad)
    .NumInputs(5)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Masked version of AdaGrad.
Concretely, given inputs (param, grad, moment, learning_rate, mask),
computes
    new_moment = moment + square(grad) if mask == 1 else 0
    effective_lr = learning_rate / (sqrt(new_moment) + epsilon)
    update = learning_rate * grad / (sqrt(new_moment) + epsilon)
    new_param = param + update if mask == 1 else 0
and returns (new_param, new_moment).

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "grad", "Gradient computed")
    .Input(3, "lr", "learning rate")
    .Input(
        4,
        "mask",
        "masks (element type is float because it helps performance "
        "and the size is not a big concern for dense parameters typically")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")

    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.");

REGISTER_CPU_OPERATOR(
    MaskedSparseAdagrad,
    MaskedSparseAdagradOp<false /*ROWWISE*/>);
OPERATOR_SCHEMA(MaskedSparseAdagrad)
    .NumInputs(6, 8)
    .NumOutputs(2, 3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Masked version of SparseAdagrad.
There are two modes of using MaskedSparseAdagrad.

The first mode is we feed mask and mask_changed inputs.
When mask_changed is 1, we apply mask to the entire param and moment, instead
of just the rows corresponding to indices. If we don't apply mask_changed when
we input a new mask, we will have a problem that rows never touched will stay
unpruned. This mode is more flexible because it allows providing arbitrary
mask that could be generated by running sophisticated offline methods.

The second mode is we provide delays and prune_ratios arguments, and let the
operator generate masks internally. This allows a quick experimentation using
simple (iterative) magnitude-based pruning. If delays and prune_ratios
arguments are present, mask and mask_changed inputs will be ignored. delays and
prune_ratios arguments should have the same lengths. For example, if delays =
[1000, 2000, 3000] and prune_ratios = [0.5, 0.7, 0.9], the operator will create
a mask for 50% sparsity at iteration 1000, a mask for 70% sparsity at iteration
2000, and so on. Whenever we change the prune_ratios, we apply the updated mask
to the entire param and moment. Otherwise, we apply the mask only the rows
accessed via indices.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Input(5, "mask", "Bit mask in uint8_t. Ignored when delay is provided")
    .Input(
        6,
        "mask_changed",
        "1 boolean value to indicate whether mask has changed")
    .Input(
        7,
        "iter",
        "Optional input to feed iteration count in int32_t. "
        "Only effective with delays and prune_ratios arguments are present")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Output(
        2,
        "mask_out",
        "Optional output to observe bit mask (in uint8_t) generated "
        "internally. Only effective with delays and prune_ratios arguments "
        "are present.")
    .Arg("epsilon", "Default 1e-5")
    .Arg("block_size", "Default 1")
    .Arg(
        "delays",
        "Optional arg to internally generate mask and control when we change "
        "prune ratios. Must be provided with prune_rates argument."
        "If present mask and mask_changed inputs are ignored")
    .Arg("prune_ratios", "Optional arg to control prune rates");

REGISTER_CPU_OPERATOR(
    MaskedRowWiseSparseAdagrad,
    MaskedSparseAdagradOp<true /*ROWWISE*/>);
OPERATOR_SCHEMA(MaskedRowWiseSparseAdagrad)
    .NumInputs(6, 9)
    .NumOutputs(2, 3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Masked version of RowWiseSparseAdagrad.
There are two modes of using MaskedRowWiseSparseAdagrad.

The first mode is we feed mask and mask_changed inputs.
When mask_changed is 1, we apply mask to the entire param and moment, instead
of just the rows corresponding to indices. If we don't apply mask_changed when
we input a new mask, we will have a problem that rows never touched will stay
unpruned. This mode is more flexible because it allows providing arbitrary
mask that could be generated by running sophisticated offline methods.

The second mode is we provide delays and prune_ratios arguments, and let the
operator generate masks internally. This allows a quick experimentation using
simple (iterative) magnitude-based pruning. If delays and prune_ratios
arguments are present, mask and mask_changed inputs will be ignored. delays and
prune_ratios arguments should have the same lengths. For example, if delays =
[1000, 2000, 3000] and prune_ratios = [0.5, 0.7, 0.9], the operator will create
a mask for 50% sparsity at iteration 1000, a mask for 70% sparsity at iteration
2000, and so on. Whenever we change the prune_ratios, we apply the updated mask
to the entire param and moment. Otherwise, we apply the mask only the rows
accessed via indices.
If iter input is present, it will be used as the iteration count. If not, the
operator will keep internal counter that will be incremented for each operator
invocation.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Input(5, "mask", "Bit mask in uint8_t. Ignored when delay is provided")
    .Input(
        6,
        "mask_changed",
        "1 boolean value to indicate whether mask has changed")
    .Input(
        7,
        "iter",
        "Optional input to feed iteration count in int32_t. "
        "Only effective with delays and prune_ratios arguments are present")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Output(
        2,
        "mask_out",
        "Optional output to observe bit mask (in uint8_t) generated "
        "internally. Only effective with delays and prune_ratios arguments "
        "are present.")
    .Arg("epsilon", "Default 1e-5")
    .Arg("block_size", "Default 1")
    .Arg(
        "delays",
        "Optional arg to internally generate mask and control when we change "
        "prune ratios. Must be provided with prune_rates argument."
        "If present mask and mask_changed inputs are ignored")
    .Arg("prune_ratios", "Optional arg to control prune rates");

} // anonymous namespace
} // namespace caffe2
