#include "parallel_batch_matmul_op.h"
#include "partition.h"

namespace caffe2 {

namespace intra_op_parallel {

ParallelBatchMatMulOp::ParallelBatchMatMulOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : ParallelOpBase(operator_def, ws),
      trans_a_(this->template GetSingleArgument<int>("trans_a", 0)),
      trans_b_(this->template GetSingleArgument<int>("trans_b", 0)),
      broadcast_(this->template GetSingleArgument<int>("broadcast", 0)) {}

ParallelBatchMatMulOp::~ParallelBatchMatMulOp() {}

bool ParallelBatchMatMulOp::RunOnDevicePrologue(int /* unused */) {
  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* Y = Output(0);

  auto ndims_A = A.ndim();
  auto dims_A = A.sizes().vec();
  auto ndims_B = B.ndim();
  auto dims_B = B.sizes().vec();

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

  typedef CPUContext Context;
  typedef float T;
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
    size_t K_dim;
    if (trans_a_) {
      M_ = dims_A[ndims_A - 1];
      K_ = dims_A[ndims_A - 2];
      K_dim = ndims_A - 2;
    } else {
      M_ = dims_A[ndims_A - 2];
      K_ = dims_A[ndims_A - 1];
      K_dim = ndims_A - 1;
    }
    if (trans_b_) {
      N_ = dims_B[ndims_B - 2];
      CAFFE_ENFORCE_EQ(
          K_,
          dims_B[ndims_B - 1],
          dimMismatchErrorString(
              K_dim, K_, ndims_B - 1, dims_B[ndims_B - 1], trans_a_, trans_b_));
    } else {
      N_ = dims_B[ndims_B - 1];
      CAFFE_ENFORCE_EQ(
          K_,
          dims_B[ndims_B - 2],
          dimMismatchErrorString(
              K_dim, K_, ndims_B - 2, dims_B[ndims_B - 2], trans_a_, trans_b_));
    }

    // Calculate output tensor shapes [B..., (M), (N)]
    // Batch dimensions will be broadcasted out to those of the longer tensor
    // A or B. Either M or N are optional if A or B, respectively are 1-D.
    std::vector<int64_t> new_dims;
    if (ndims_A >= ndims_B) {
      new_dims.assign(dims_A.begin(), dims_A.end() - 2);
    } else {
      new_dims.assign(dims_B.begin(), dims_B.end() - 2);
    }
    if (!A_broadcasted) {
      new_dims.push_back(M_);
    } else {
      new_dims.push_back(1);
    }
    if (!B_broadcasted) {
      new_dims.push_back(N_);
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
    A_stride_ = 1; // How far to increment A pointer each itr
    B_stride_ = 1; // How far to increment B pointer each itr
    Y_stride_ = 1; // How far to increment Y pointer each itr
    // How many "inner batches" we have. That is, the product of sizes for
    // the slices excluding M, K, and N, for their respective matrices.
    num_sub_batches_ = 1;
    if (ndims_A >= ndims_B) {
      auto first_r_itr = dims_A.rbegin();
      auto output_r_itr = new_dims.rbegin();
      for (size_t i = 0; i < num_inner_dims; ++i) {
        A_stride_ *= *(first_r_itr + i);
        Y_stride_ *= *(output_r_itr + i);
        if (i >= 2) {
          num_sub_batches_ *= *(first_r_itr + i);
        }
      }
      B_stride_ = 0;
    } else {
      A_stride_ = 0;
      auto second_r_itr = dims_B.rbegin();
      auto output_r_itr = new_dims.rbegin();
      for (size_t i = 0; i < num_inner_dims; ++i) {
        B_stride_ *= *(second_r_itr + i);
        Y_stride_ *= *(output_r_itr + i);
        if (i >= 2) {
          num_sub_batches_ *= *(second_r_itr + i);
        }
      }
    }

    num_outer_batches_ = 1;
    for (size_t i = 0; i < num_outer_dims; ++i) {
      num_outer_batches_ *= new_dims[i];
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
    Y->template mutable_data<T>();
  }
  return true;
}

bool ParallelBatchMatMulOp::RunOnDeviceParallel(int task_id, int num_tasks) {
  typedef float T;
  typedef CPUContext Context;
  typedef DefaultEngine Engine;

  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* Y = Output(0);

  auto ndims_A = A.ndim();
  auto dims_A = A.sizes().vec();
  auto ndims_B = B.ndim();
  auto dims_B = B.sizes().vec();

  if (ndims_A == 1 && ndims_B == 1) {
    return true;
  }

  // Zero batch dimension indicates no elements
  if (num_sub_batches_ == 0 || num_outer_batches_ == 0) {
    return true;
  }

  auto* data_A = A.template data<T>();
  auto* data_B = B.template data<T>();
  auto* Y_data = Y->template mutable_data<T>();

  size_t sub_batch_begin, sub_batch_end;
  std::tie(sub_batch_begin, sub_batch_end) =
      Get1DPartition(num_sub_batches_, num_tasks, task_id);

  // TODO(T23893772): doing this in a loop is likely going to be slow on GPU
  for (size_t p = 0; p < num_outer_batches_; ++p) {
    math::GemmStridedBatched<T, Context, Engine>(
        trans_a_ ? CblasTrans : CblasNoTrans,
        trans_b_ ? CblasTrans : CblasNoTrans,
        sub_batch_end - sub_batch_begin,
        M_,
        N_,
        K_,
        1.0f,
        data_A + p * A_stride_ + sub_batch_begin * M_ * K_,
        M_ * K_,
        data_B + p * B_stride_ + sub_batch_begin * K_ * N_,
        K_ * N_,
        0.0f,
        Y_data + p * Y_stride_ + sub_batch_begin * M_ * N_,
        M_ * N_,
        &context_);
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    INTRA_OP_PARALLEL,
    ParallelBatchMatMulOp);

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB
REGISTER_CPU_OPERATOR_WITH_ENGINE(BatchMatMul, TBB, ParallelBatchMatMulOp);
#endif

} // namespace intra_op_parallel

} // namespace caffe2
