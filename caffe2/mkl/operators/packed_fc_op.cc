#include <cstdint>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/math.h"

#ifdef CAFFE2_HAS_MKL_SGEMM_PACK

namespace caffe2 {

CAFFE_KNOWN_TYPE(mkl::MKLPackedMatrix);

namespace mkl {

class PackedFCOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  PackedFCOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)) {
    OPERATOR_NEEDS_FEATURE(
        GetCpuId().avx2() || operator_def.type() == "PackedFC",
        "If you are trying to use PackedFCOp as a FC with PACKED engine on "
        "a machine that does not have avx2, be noted that the functionality "
        "is not tuned and you are better off directly using FC.");
    // TODO(jiayq): after MKL update, remove this constraint. This is different
    // from the check above, as the above is a performance hint and the below
    // is about correctness.
    CAFFE_ENFORCE(
        GetCpuId().avx2(),
        "Do not run PackedFC on a machine that does not have avx2 "
        "right now, as there is a known issue with MKL 2017.0.098 "
        "that produces wrong results on non-avx2 machines.");
  }
  ~PackedFCOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& b = Input(2);
    auto* Y = Output(0);
    CAFFE_ENFORCE(b.ndim() == 1, b.ndim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int K = X.size_from_dim(canonical_axis);
    const int N = b.size();

    // Check out what is the passed in format.
    const MKLPackedMatrix* packed_matrix = nullptr;
    if (OperatorBase::InputIsType<TensorCPU>(1)) {
      const auto& W = Input(1);
      CAFFE_ENFORCE_EQ(W.ndim(), 2);
      CAFFE_ENFORCE_EQ(W.dim32(0), N);
      CAFFE_ENFORCE_EQ(W.dim32(1), K);
      // Note(jiayq): This will strictly check that we have a proper usage of
      // the PackedFC operator. The motivation is that, this operator is
      // stateful unlike most ops in Caffe2, but checking whether the weight
      // has changed matters quite a lot in the critical path. We only enable
      // this test during DEBUG mode for performance considerations.
      DCHECK(hash_ == 0 || hash_ == Hash(W.template data<float>(), W.size()))
          << "PackedFCOp is currently stateful: you should not change the "
             "weight during runtime. This is only sanity-checked in debug "
             "mode for speed considerations.";
      if (!local_packed_matrix_.get() || local_packed_matrix_->n_ != M) {
        // If there is no pre packed matrix, or the batch size changed, we
        // do a re-pack.
        local_packed_matrix_.reset(new MKLPackedMatrix(
            CblasBMatrix,
            CblasTrans,
            M,
            N,
            K,
            1.f,
            W.template data<float>(),
            K));
      }
      packed_matrix = local_packed_matrix_.get();
    } else if (OperatorBase::InputIsType<MKLPackedMatrix>(1)) {
      packed_matrix = &OperatorBase::Input<MKLPackedMatrix>(1);
    }
    CAFFE_ENFORCE_EQ(packed_matrix->m_, M);
    CAFFE_ENFORCE_EQ(packed_matrix->k_, K);
    CAFFE_ENFORCE_EQ(packed_matrix->n_, N);
    // Do we want to check the other flags as well?

    Y_shape_cache_ = X.dims();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    Y->Resize(Y_shape_cache_);
    CAFFE_ENFORCE(M * N == Y->size());

    cblas_sgemm_compute(
        CblasRowMajor,
        CblasNoTrans,
        CblasPacked,
        M,
        N,
        K,
        X.template data<float>(),
        K,
        packed_matrix->data_,
        K,
        0,
        Y->template mutable_data<float>(),
        N);

    // Add bias term
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<float, CPUContext>(
          M, 1.f, bias_multiplier_.template mutable_data<float>(), &context_);
    }
    math::Gemm<float, CPUContext>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier_.template data<float>(),
        b.template data<float>(),
        1,
        Y->template mutable_data<float>(),
        &context_);
    return true;
  }

 protected:
  uint32_t Hash(const float* ptr, size_t n) {
    uint32_t hash = 0;
    const uint32_t* ptr_i = reinterpret_cast<const uint32_t*>(ptr);
    for (int i = 0; i < n; ++i) {
      hash ^= ptr_i[i];
    }
    return hash;
  }
  size_t axis_{1};
  uint32_t hash_{0};
  vector<TIndex> Y_shape_cache_;
  Tensor<CPUContext> bias_multiplier_;
  std::unique_ptr<MKLPackedMatrix> local_packed_matrix_;
};

} // namespace mkl

REGISTER_CPU_OPERATOR(PackedFC, mkl::PackedFCOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, PACKED, mkl::PackedFCOp);

OPERATOR_SCHEMA(PackedFC).NumInputs(3).NumOutputs(1).SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully connected
layer with 2D weight matrix W and 1D bias vector b. This is essentially the
same as the FC operator but allows one to pack the weight matrix for more
efficient inference. See the schema for the FC op for details.

Unlike many other operators in Caffe2, this operator is stateful: it assumes
that the input weight matrix W never changes, so it is only suitable for
inference time when the weight matrix never gets updated by any other ops.
Due to performance considerations, this is not checked in non-debug builds.
)DOC");

SHOULD_NOT_DO_GRADIENT(PackedFC);
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_SGEMM_PACK
