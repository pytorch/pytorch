#ifndef CAFFE2_OPERATORS_MATMUL_OP_H_
#define CAFFE2_OPERATORS_MATMUL_OP_H_

#include <sstream>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context, class Engine = DefaultEngine>
class BatchMatMulOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchMatMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        trans_a_(OperatorBase::GetSingleArgument<int>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int>("trans_b", 0)),
        broadcast_(OperatorBase::GetSingleArgument<int>("broadcast", 0)),
        use_scratch_(OperatorBase::GetSingleArgument<int>("use_scratch", 0)) {
    if (use_scratch_) {
      scratch_ = std::make_shared<Tensor<Context>>();
    }
  }

  ~BatchMatMulOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* Y = Output(0);

    auto ndims_A = A.ndim();
    auto dims_A = A.dims();
    auto ndims_B = B.ndim();
    auto dims_B = B.dims();

    auto noBroadcastErrorMsg = [](size_t dim1, size_t dim2) {
      std::stringstream ss;
      ss << "Inputs with dimensions A = ";
      ss << dim1;
      ss << " and B = ";
      ss << dim2;
      ss << " is not supported with broadcast=0. Did you forget to set the "
            "broadcast flag?";
      return ss.str();
    };

    // These should all be false if we're not broadcasting.
    bool dimMismatch = ndims_A != ndims_B;
    bool dimsLessThan1D = ndims_A < 2;
    CAFFE_ENFORCE(
        broadcast_ || (!dimMismatch && !dimsLessThan1D),
        noBroadcastErrorMsg(ndims_A, ndims_B));

    auto* data_A = A.template data<T>();
    auto* data_B = B.template data<T>();

    auto dimMismatchErrorString = [](size_t dimnum1,
                                     size_t dim1,
                                     size_t dimnum2,
                                     size_t dim2,
                                     bool trans_a,
                                     bool trans_b) {
      std::stringstream ss;
      ss << "Expected dimension ";
      ss << dimnum1;
      ss << " of tensor A with value ";
      ss << dim1;
      ss << " to match dimension ";
      ss << dimnum2;
      ss << " of tensor B with value ";
      ss << dim2;
      ss << ". trans_a = ";
      ss << trans_a;
      ss << " trans_b = ";
      ss << trans_b;
      return ss.str();
    };

    if (ndims_A == 1 && ndims_B == 1) {
      // vector-vector
      CAFFE_ENFORCE_EQ(
          dims_A[0],
          dims_B[0],
          "Vector-vector product requires each of the vectors to "
          "be the same size.");
      Y->Resize(1);
      math::Dot<T, Context>(
          dims_A[0], data_A, data_B, Y->template mutable_data<T>(), &context_);
    } else {
      bool A_broadcasted = false, B_broadcasted = false;
      if (ndims_A == 1) {
        dims_A.insert(dims_A.begin(), 1);
        ndims_A = 2;
        A_broadcasted = true;
      }
      if (ndims_B == 1) {
        dims_B.push_back(1);
        ndims_B = 2;
        B_broadcasted = true;
      }
      // matrix-matrix with batches
      // [B1..., M, K] * [B2..., K, N] -> [B..., M, N]
      // In the event that A or B are one-dimensional, the trailing or leading
      // 1 is not added to the output tensor's size.

      // First step: partition the tensors into inner and outer blocks.
      // Ignoring the last two dimensions of A and B, ensure that one of the
      // tensors' dimensions is a suffix of the other. For example,
      // [4, x, x] is a suffix of [2, 3, 4, x, x]. In this example, the
      // dimensions of size 2 and 3 will be broadcasted, so we partition into
      // 2*3=6 individual instances of batched GEMM with A and B \in [4, x, x].
      size_t num_inner_dims = std::min(ndims_A, ndims_B);
      for (size_t i = 2; i < num_inner_dims; ++i) {
        auto first_r_itr = dims_A.rbegin();
        auto second_r_itr = dims_B.rbegin();
        CAFFE_ENFORCE_EQ(
            *(first_r_itr + i),
            *(second_r_itr + i),
            dimMismatchErrorString(
                ndims_A - i - 1,
                *(first_r_itr + i),
                ndims_B - i - 1,
                *(second_r_itr + i),
                trans_a_,
                trans_b_));
      }
      size_t num_outer_dims = std::max(ndims_A, ndims_B) - num_inner_dims;

      // Standard M, N, and K parameters respecting GEMM API and transpose
      // flags
      size_t M, N, K, K_dim;
      if (trans_a_) {
        M = dims_A[ndims_A - 1];
        K = dims_A[ndims_A - 2];
        K_dim = ndims_A - 2;
      } else {
        M = dims_A[ndims_A - 2];
        K = dims_A[ndims_A - 1];
        K_dim = ndims_A - 1;
      }
      if (trans_b_) {
        N = dims_B[ndims_B - 2];
        CAFFE_ENFORCE_EQ(
            K,
            dims_B[ndims_B - 1],
            dimMismatchErrorString(
                K_dim,
                K,
                ndims_B - 1,
                dims_B[ndims_B - 1],
                trans_a_,
                trans_b_));
      } else {
        N = dims_B[ndims_B - 1];
        CAFFE_ENFORCE_EQ(
            K,
            dims_B[ndims_B - 2],
            dimMismatchErrorString(
                K_dim,
                K,
                ndims_B - 2,
                dims_B[ndims_B - 2],
                trans_a_,
                trans_b_));
      }

      // Calculate output tensor shapes [B..., (M), (N)]
      // Batch dimensions will be broadcasted out to those of the longer tensor
      // A or B. Either M or N are optional if A or B, respectively are 1-D.
      std::vector<TIndex> new_dims;
      if (ndims_A >= ndims_B) {
        new_dims.assign(dims_A.begin(), dims_A.end() - 2);
      } else {
        new_dims.assign(dims_B.begin(), dims_B.end() - 2);
      }
      if (!A_broadcasted) {
        new_dims.push_back(M);
      } else {
        new_dims.push_back(1);
      }
      if (!B_broadcasted) {
        new_dims.push_back(N);
      } else {
        new_dims.push_back(1);
      }

      // Calculate strides. Continuing our example above,
      //   [4, M, K] * [2, 3, 4, K, N] = [2, 3, 4, M, N]
      // We calculate this as follows:
      //   1) Treat the outer batch dimensions as flattened, i.e. view the B
      //      tensor here as [6, 4, K, N] and Y as [6, 4, M, N]. The same rea-
      //      soning is analogous for the case where # dims A >= # dims B.
      //   2) Perform this operation:
      //        for i in range(6):
      //          Y[i, :, :, :] = BatchMatMul(A, B[i, :, :, :])
      size_t A_stride = 1; // How far to increment A pointer each itr
      size_t B_stride = 1; // How far to increment B pointer each itr
      size_t Y_stride = 1; // How far to increment Y pointer each itr
      // How many "inner batches" we have. That is, the product of sizes for
      // the slices excluding M, K, and N, for their respective matrices.
      size_t num_sub_batches = 1;
      if (ndims_A >= ndims_B) {
        auto first_r_itr = dims_A.rbegin();
        auto output_r_itr = new_dims.rbegin();
        for (size_t i = 0; i < num_inner_dims; ++i) {
          A_stride *= *(first_r_itr + i);
          Y_stride *= *(output_r_itr + i);
          if (i >= 2) {
            num_sub_batches *= *(first_r_itr + i);
          }
        }
        B_stride = 0;
      } else {
        A_stride = 0;
        auto second_r_itr = dims_B.rbegin();
        auto output_r_itr = new_dims.rbegin();
        for (size_t i = 0; i < num_inner_dims; ++i) {
          B_stride *= *(second_r_itr + i);
          Y_stride *= *(output_r_itr + i);
          if (i >= 2) {
            num_sub_batches *= *(second_r_itr + i);
          }
        }
      }

      size_t num_outer_batches = 1;
      for (size_t i = 0; i < num_outer_dims; ++i) {
        num_outer_batches *= new_dims[i];
      }

      // Mutually exclusive since otherwise we would've taken the vector-vector
      // path above
      if (A_broadcasted) {
        new_dims.erase(new_dims.end() - 2);
      } else if (B_broadcasted) {
        new_dims.erase(new_dims.end() - 1);
      }

      // Allocate output tensor
      Y->Resize(new_dims);
      auto* Y_data = Y->template mutable_data<T>();

      // Zero batch dimension indicates no elements
      if (num_sub_batches == 0 || num_outer_batches == 0) {
        return true;
      }

      // TODO(T23893772): doing this in a loop is likely going to be slow on GPU
      for (size_t p = 0; p < num_outer_batches; ++p) {
        math::GemmBatched<T, Context, Engine>(
            trans_a_ ? CblasTrans : CblasNoTrans,
            trans_b_ ? CblasTrans : CblasNoTrans,
            num_sub_batches,
            M,
            N,
            K,
            1.0f,
            data_A + p * A_stride,
            data_B + p * B_stride,
            0.0f,
            Y_data + p * Y_stride,
            &context_,
            use_scratch_ ? scratch_.get() : nullptr);
      }
    }
    return true;
  }

 protected:
  bool trans_a_;
  bool trans_b_;
  bool broadcast_;

  bool use_scratch_;
  std::shared_ptr<Tensor<Context>> scratch_;
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_MATMUL_OP_H_ */
