#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>

#include <oneDNN/oneDNN.h>
// #include <runtime/Utils.h>
// #include <utils/oneMKLUtils.h>
#include <vector>

// #include "comm/ATDispatch.h"
// #include "comm/RegistrationDeclarations.h"

#include <c10/util/typeid.h>

// using namespace dnnl;
// using namespace xpu::dpcpp;
// using namespace xpu::oneDNN;

namespace at {
namespace xpu {
namespace impl {

static inline bool check_broadcast(
    const Tensor& src,
    const IntArrayRef& shape) {
  auto src_dim = src.dim();
  auto tgt_dim = shape.size();
  if (src_dim == 0 && src_dim < tgt_dim)
    return true;
  if (src_dim > tgt_dim)
    return false;
  do {
    src_dim--;
    tgt_dim--;
    auto size = src.size(src_dim);
    if (size != 1 && size != shape[tgt_dim])
      return false;
  } while (src_dim);
  return true;
}

#ifdef USE_ONEMKL
template <typename scalar_t>
static void gemm_batch(
    sycl::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    scalar_t alpha,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    scalar_t* b,
    int64_t ldb,
    int64_t stride_b,
    scalar_t beta,
    scalar_t* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      stride_a,
      b,
      ldb,
      stride_b,
      beta,
      c,
      ldc,
      stride_c,
      batch_size);
}

template <>
void gemm_batch<c10::complex<double>>(
    sycl::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    c10::complex<double> alpha,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<double>* b,
    int64_t ldb,
    int64_t stride_b,
    c10::complex<double> beta,
    c10::complex<double>* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      *reinterpret_cast<std::complex<double>*>(&alpha),
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<double>*>(b),
      ldb,
      stride_b,
      *reinterpret_cast<std::complex<double>*>(&beta),
      reinterpret_cast<std::complex<double>*>(c),
      ldc,
      stride_c,
      batch_size);
}

template <>
void gemm_batch<c10::complex<float>>(
    sycl::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    c10::complex<float> alpha,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<float>* b,
    int64_t ldb,
    int64_t stride_b,
    c10::complex<float> beta,
    c10::complex<float>* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      *reinterpret_cast<std::complex<float>*>(&alpha),
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<float>*>(b),
      ldb,
      stride_b,
      *reinterpret_cast<std::complex<float>*>(&beta),
      reinterpret_cast<std::complex<float>*>(c),
      ldc,
      stride_c,
      batch_size);
}
#endif

static void mkl_baddbmm(
    Tensor& result,
    const Tensor& self,
    Tensor batch1,
    Tensor batch2,
    const Scalar& beta,
    const Scalar& alpha) {
#ifdef USE_ONEMKL
  // colum major
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  auto batch1_sizes = batch1.sizes();
  auto batch2_sizes = batch2.sizes();
  auto batch1_strides = batch1.strides();
  auto batch2_strides = batch2.strides();

  TORCH_CHECK(
      batch2_sizes[0] == batch1_sizes[0] && batch2_sizes[1] == batch1_sizes[2],
      "Expected size for first two dimensions of batch2 tensor to be: [",
      batch1_sizes[0],
      ", ",
      batch1_sizes[2],
      "] but got: [",
      batch2_sizes[0],
      ", ",
      batch2_sizes[1],
      "].");

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    auto b_self = expand_size(
        self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "mkl_matmul");
    result.resize_as_(*b_self).copy_(*b_self);
  } else {
    // For mkl_baddbmm, have to convert it to contiguous format(only update meta
    // data, and don't copy memory) for such kind of tensor below: E.g.: the
    // tensor whose size is [10, 12, 50], and stride is [50, 500, 1], where
    // oneMKL lib cannot handle this kind of stride. Because stridec from oneMKL
    // strided style API means step size for each sample in the same batch.
    // However, for mkl_matmul, the stridec is always c.numel(), because we only
    // have 1 sample when we do addmm.
    result.resize_(
        {batch1.size(0), batch1.size(1), batch2.size(2)},
        at::MemoryFormat::Contiguous);
  }

  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      result.zero_();
    }
  }

  bool transpose_c = false;
  Tensor c;

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) ||
       (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    // colum major
    transpose_c = false;
    c = result.resolve_conj();
  } else if (
      (result_strides[2] == 1) &&
      (result_sizes[1] == 1 ||
       (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    // row major
    std::swap(batch1, batch2);
    std::swap(batch1_sizes, batch2_sizes);
    std::swap(batch1_strides, batch2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    c = result.resolve_conj().transpose(1, 2).contiguous().transpose_(1, 2);
  }

  const int64_t m = result_sizes[transpose_c ? 2 : 1];
  const int64_t n = result_sizes[transpose_c ? 1 : 2];
  const int64_t k = batch1_sizes[transpose_c ? 1 : 2];

  // Cast batch1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (batch1_strides[transpose_c ? 2 : 1] == 1 &&
      batch1_strides[transpose_c ? 1 : 2] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = batch1.resolve_conj();
  } else if (
      batch1_strides[transpose_c ? 1 : 2] == 1 &&
      batch1_strides[transpose_c ? 2 : 1] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = batch1;
  } else {
    transpose_a = !transpose_c;
    a = batch1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast batch2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (batch2_strides[transpose_c ? 2 : 1] == 1 &&
      batch2_strides[transpose_c ? 1 : 2] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = batch2.resolve_conj();
  } else if (
      batch2_strides[transpose_c ? 1 : 2] == 1 &&
      batch2_strides[transpose_c ? 2 : 1] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = batch2;
  } else {
    transpose_b = !transpose_c;
    b = batch2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 2 : 1];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 2 : 1];
  // for the corner case: result tensor with size [b, m, 1], stride [m, 1, 1]
  // we cannot use stride to get its leading dimension, whose value should be m.
  int64_t ldc;
  if (c.strides()[1] == c.strides()[2] == 1) {
    ldc = c.sizes()[transpose_c ? 2 : 1];
  } else {
    ldc = c.strides()[transpose_c ? 1 : 2];
  }

  const int64_t stridea = a.strides()[0];
  const int64_t strideb = b.strides()[0];
  const int64_t stridec = c.strides()[0];
  int64_t num_batch = c.sizes()[0];

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "mkl_baddbmm", [&] {
        gemm_batch<scalar_t>(
            dpcpp_queue,
            transpose_a ? a.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N, // nontrans = 0, trans = 1,
                                                     // conjtrans = 3,
            transpose_b ? b.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N,
            m,
            n,
            k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(),
            lda,
            stridea,
            b.data_ptr<scalar_t>(),
            ldb,
            strideb,
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(),
            ldc,
            stridec,
            num_batch);
      });

  if (!result.is_same(c)) {
    result.copy_(c);
  }
#endif
}

static void mkl_matmul(
    Tensor& result,
    const Tensor& self,
    Tensor m1,
    Tensor m2,
    Scalar beta,
    Scalar alpha) {
#ifdef USE_ONEMKL
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    auto b_self = expand_size(self, {m1_sizes[0], m2_sizes[1]}, "mkl_matmul");
    result.resize_as_(*b_self).copy_(*b_self);
  } else {
    result.resize_({m1_sizes[0], m2_sizes[1]});
  }

  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  }

  bool transpose_c = false;
  Tensor c;

  // Cast result as matrix a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 ||
       result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (
      result_strides[1] == 1 &&
      (result_sizes[0] == 1 ||
       result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  // Cast m1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (
      m1_strides[transpose_c ? 0 : 1] == 1 &&
      m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast m2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (
      m2_strides[transpose_c ? 0 : 1] == 1 &&
      m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  // for the corner case: result tensor with size [m, 1], stride [1, 1]
  // we cannot use stride to get its leading dimension, whose value should be m.
  int64_t ldc;
  if (1 == c.strides()[0] == c.strides()[1]) {
    ldc = c.sizes()[transpose_c ? 1 : 0];
  } else {
    ldc = c.strides()[transpose_c ? 0 : 1];
  }

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  // use colum major
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "mkl_matmul", [&] {
        gemm_batch<scalar_t>(
            dpcpp_queue,
            transpose_a ? a.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N, // nontrans = 0, trans = 1,
                                                     // conjtrans = 3,
            transpose_b ? b.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N,
            m,
            n,
            k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(),
            lda,
            a.numel(),
            b.data_ptr<scalar_t>(),
            ldb,
            b.numel(),
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(),
            ldc,
            c.numel(),
            1);
      });

  if (!c.is_same(result)) {
    result.copy_(c);
  }
#endif
}

/***** The helper function to get post binary(or sum) for onednn_matmul *****
In onednn, it supports: result = BinaryOP(alpha * (m1 @ m2 + bias), beta *
binary). Since the inputs/outputs shapes of Matmul are complicated,
this helper function is used to adjust binary tensor size according different
matmul cases.*/
static bool get_onednn_matmul_binary_attr(
    Tensor& result,
    xpu::onednn::Attr& attr,
    int dim_tensor1,
    int dim_tensor2,
    DimVector output_shape,
    bool t2_is_matrix = true,
    bool should_fold_tensor1 = false,
    bool should_fold_tensor2 = false) {
  xpu::onednn::Attr attr_update;
  for (int i = 0; i < attr.ops_params_.size(); ++i) {
    xpu::onednn::kind_t kind = attr.ops_params_[i].kind_;
    if (kind != xpu::onednn::kind_t::binary || !attr.ops_params_[i].binary_.defined()) {
      attr_update.ops_params_.push_back(attr.ops_params_[i]);
      continue;
    }

    float beta = attr.ops_params_[i].scale_;
    bool need_binary = attr.ops_params_[i].binary_.defined() && (beta != 0.f);
    if (!need_binary) {
      continue;
    }

    Tensor binary_final;
    std::vector<int64_t> compute_shape = result.sizes().vec();
    if (dim_tensor1 == 2 && dim_tensor2 == 1) {
      // case 2
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      binary_final = binary->unsqueeze(1);
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
      // case 3
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      binary_final = binary->unsqueeze(0);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
      // case 4
      const auto binary =
          MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      if (binary->dim() < output_shape.size())
        binary_final = binary->unsqueeze(0);
      else
        binary_final = *binary;
    } else if (should_fold_tensor1) {
      // case 5
      auto binary = t2_is_matrix
          ? MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_)
          : MaybeOwned<Tensor>::owned(
                attr.ops_params_[i].binary_.unsqueeze(-1));
      while (binary->dim() < output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() >= compute_shape.size()) {
        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 1);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->contiguous().view(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          std::vector<int64_t> acc_shape = {compute_shape[0], shape_fold[1]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }

    } else if (should_fold_tensor2) {
      // case 6
      auto binary = t2_is_matrix
          ? MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_)
          : MaybeOwned<Tensor>::owned(
                attr.ops_params_[i].binary_.unsqueeze(-1));

      while (binary->dim() < output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() >= compute_shape.size()) {
        if (t2_is_matrix)
          binary = MaybeOwned<Tensor>::owned(binary->mT());

        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 1);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->contiguous().view(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          std::vector<int64_t> acc_shape = {compute_shape[0], shape_fold[1]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }
    } else {
      // case 7
      auto binary = MaybeOwned<Tensor>::borrowed(attr.ops_params_[i].binary_);
      while (binary->dim() < output_shape.size())
        binary = MaybeOwned<Tensor>::owned(binary->unsqueeze(0));

      if (binary->dim() > 3) {
        std::vector<int64_t> shape = binary->sizes().vec();
        auto shape_fold = DimVector(shape.begin(), shape.end() - 2);
        const auto first_dim = c10::multiply_integers(shape_fold);
        shape_fold = {first_dim, *(shape.end() - 2), *(shape.end() - 1)};
        if (first_dim == compute_shape[0] || first_dim == 1) {
          binary_final = binary->reshape(shape_fold);
        } else {
          auto expand_shape = output_shape;
          expand_shape[expand_shape.size() - 1] =
              shape_fold[shape_fold.size() - 1];
          expand_shape[expand_shape.size() - 2] =
              shape_fold[shape_fold.size() - 2];
          std::vector<int64_t> acc_shape = {
              compute_shape[0], shape_fold[1], shape_fold[2]};
          binary_final =
              binary->expand(expand_shape).contiguous().view(acc_shape);
        }
      } else {
        binary_final = *binary;
      }
    }

    if (!xpu::onednn::binary_valid(result, binary_final, true)) {
      attr = xpu::onednn::Attr();
      return false;
    }

    auto algo = attr.ops_params_[i].algo_;
    if (algo == attr.kind_with_binary_add) {
      // result = (m1 x m2) + beta * binary
      // result = beta * (1.f / beta * (mat1 * mat2) + binary)
      // Since oneDNN only supports sum_scale=1.0 for non-int8 case,
      // we do this formula transformation.
      if (beta != 1.f) {
        attr_update.append_post_eltwise(
            1.f, 1.f / beta, 0.f, attr.kind_with_linear);
      }
      attr_update.append_post_binary(algo, binary_final);
      if (beta != 1.f) {
        attr_update.append_post_eltwise(1.f, beta, 0.f, attr.kind_with_linear);
      }
    } else {
      // binary_mul: result = (m1 x m2) * binary * beta;
      // binary_div: result = (m1 x m2) / binary * beta;
      // binary_min: result = min((m1 x m2), binary) * beta;
      // binary_max: result = max((m1 x m2), binary) * beta;
      // binary_eq: result = eq((m1 x m2), binary) * beta;
      // binary_ne: result = ne((m1 x m2), binary) * beta;
      // binary_ge: result = ge((m1 x m2), binary) * beta;
      // binary_gt: result = gt((m1 x m2), binary) * beta;
      // binary_le: result = le((m1 x m2), binary) * beta;
      // binary_lt: result = lt((m1 x m2), binary) * beta;
      attr_update.append_post_binary(algo, binary_final);
      if (beta != 1.f)
        attr_update.append_post_eltwise(1.f, beta, 0.f, attr.kind_with_linear);
    }
  }
  attr_update.q_scale_ = attr.q_scale_;
  attr_update.q_zero_point_ = attr.q_zero_point_;
  attr = attr_update;
  return true;
}

static bool should_fold(const Tensor& tensor1, const int64_t dim_tensor2) {
  const auto dim_tensor1 = tensor1.dim();
  if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    const auto t1_sizes_ptr = tensor1.sizes().cbegin();
    const auto t1_strides = tensor1.strides();
    if (dim_tensor1 == 3 && dim_tensor2 == 2 && t1_strides.back() != 1 &&
        t1_strides.front() == t1_sizes_ptr[1] * t1_sizes_ptr[2]) {
      // First dim is slowest moving, and then the following two dims are //
      // transposed. This can happen for example by permute(0, 2, 1).      //
      // First 2 dims could be folded to use mm but would require permutation //
      // with actual data movement, which can be instead handled by BMM with
      // each      // GEMM transposed. This can be generalized to a tensor with
      // dim X + Y + Z where X, Y, and Z      // dims are contiguous, Y dims and
      // Z dims are transposed, and X, Y, Z > 0.      // For example, this can
      // happen by permute(0, 1, 5, 2, 3, 4), where X = 2, Y = 3, and Z = 1.
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

// Port from PyTorch _matmul_impl function
/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, (1d) the dot product (scalar) is
returned.
- If the arguments are 2D - 1D or 1D - 2D, the matrix-vector product is
returned.
- If both arguments are 2D, the matrix-matrix product is returned.
- If one of the arguments is ND with N >= 3 and the other is 1D or 2D, and
some conditions on the strides apply (see should_fold) we fold the first N-1
dimensions of the ND argument to form a matrix, call mm or mv, reshape it
back to ND and return it
- Otherwise, we return bmm, after broadcasting and folding the batched
dimensions if there's more than one
*/
static Tensor& matmul_fusion_variants(
    Tensor& output,
    const Tensor& tensor1,
    const Tensor& tensor2,
    bool trans,
    xpu::onednn::Attr& attr,
    bool& is_fused,
    Tensor bias = at::Tensor()) {
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();
  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(
      dim_tensor1 != 0 && dim_tensor2 != 0,
      "both arguments to matmul need to be at least 1D, but they are ",
      dim_tensor1,
      "D and ",
      dim_tensor2,
      "D");

  bool should_fold_tensor1 = should_fold(tensor1, dim_tensor2);
  bool should_fold_tensor2 = should_fold(tensor2, dim_tensor1);

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    // case1:
    // original size: [6] x [6] -> []
    is_fused = true;
    Tensor result = output.defined() ? output.view({1, 1})
                                     : at::empty({1, 1}, tensor1.options());
    xpu::onednn::matmul(
        result,
        tensor1.view({1, tensor1.size(0)}),
        tensor2.view({tensor2.size(0), 1}),
        bias,
        trans,
        attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result;
      output.resize_({});
    }
    return output;
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    // case2:
    // original sizes: [4, 2] x [2] -> [4]
    // onednn sizes: [4, 2] x [2, 1] -> [4, 1]
    DimVector output_shape({tensor1.size(0)});
    DimVector result_shape({tensor1.size(0), 1});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());
    Tensor t2 = tensor2.view({tensor2.size(0), 1});

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    xpu::onednn::matmul(result, tensor1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result.view(output_shape);
    }
    return output;
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    // case3:
    // original sizes: [2] x [2, 6] -> [6]
    // onednn sizes: [1, 2] x [2, 6] -> [1, 6]
    DimVector output_shape({tensor2.size(1)});
    if (!trans)
      output_shape[0] = tensor2.size(0);
    Tensor t1 = tensor1.unsqueeze(0);
    DimVector result_shape({1, output_shape[0]});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    xpu::onednn::matmul(result, t1, tensor2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result.view(output_shape);
    }
    return output;
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // case4:
    // original sizes: [4, 2] x [2, 6] -> [4, 6]
    // onednn sizes: [4, 2] x [2, 6] -> [4, 6]
    DimVector output_shape({tensor1.size(0), tensor2.size(1)});
    if (!trans)
      output_shape[1] = tensor2.size(0);

    Tensor result =
        output.defined() ? output : at::empty(output_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    xpu::onednn::matmul(result, tensor1, tensor2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result;
    }
    return output;
  } else if (should_fold_tensor1) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2)
    // case5-1:
    // original sizes: [3, 4, 2] x [2, 6] -> [3, 4, 6]
    // onednn sizes: [12, 2] x [2, 6] -> [12, 6]
    // case5-2:
    // original sizes: [3, 4, 2] x [2] -> [3, 4]
    // onednn sizes: [12, 2] x [2, 1] -> [12, 1]
    const auto t1_own = MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2_own = MaybeOwned<Tensor>::borrowed(tensor2);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    if (trans)
      output_shape.push_back(t2.size(1));
    else
      output_shape.push_back(t2.size(0));
    DimVector result_shape({t1.size(0), output_shape[output_shape.size() - 1]});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());
    is_fused = get_onednn_matmul_binary_attr(
        result,
        attr,
        dim_tensor1,
        dim_tensor2,
        output_shape,
        t2_is_matrix,
        should_fold_tensor1,
        should_fold_tensor2);
    xpu::onednn::matmul(result, t1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    output = t2_is_matrix ? output : output.squeeze(-1);
    return output;
  } else if (should_fold_tensor2) {
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // case6-1:
    // original sizes: [2] x [3, 2, 4] = [3, 4]
    // onednn sizes: [12, 2] x [2, 1] = [12, 1]
    // or
    // original sizes: [2] x [2, 3, 2, 4] = [2, 3, 4]
    // onednn sizes: [24, 2] x [2, 1] = [24, 1]

    // case6-2:
    // original sizes: [6, 2] x [3, 2, 4] = [3, 6, 4]
    // onednn sizes: [12, 2] x [2, 6] = [12, 6]
    // or
    // original sizes: [6, 2] x [2, 3, 2, 4] = [2, 3, 6, 4]
    // onednn sizes: [24, 2] x [2, 6] = [24, 6]

    const auto t1_own = trans
        ? MaybeOwned<Tensor>::owned(tensor2.mT())
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2).mT());
    trans = true;
    const auto t2_own = dim_tensor1 == 2
        ? MaybeOwned<Tensor>::owned(tensor1.t())
        : MaybeOwned<Tensor>::borrowed(tensor1);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    output_shape.push_back(t2.size(1));
    DimVector result_shape({t1.size(0), t2.size(1)});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result,
        attr,
        dim_tensor1,
        dim_tensor2,
        output_shape,
        t2_is_matrix,
        should_fold_tensor1,
        should_fold_tensor2);
    xpu::onednn::matmul(result, t1, t2, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    output = t2_is_matrix ? output.mT().contiguous() : output.squeeze(-1);
    return output;
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // case7-1:
    // original sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // onednn sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // case7-2:
    // original sizes: [5, 1, 4, 2] x [3, 2, 6] = [5, 3, 4, 6]
    // onednn sizes: [15, 4, 2] x [15, 2, 6] = [15, 4, 6]
    const auto t2_own = trans
        ? MaybeOwned<Tensor>::borrowed(tensor2)
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2));
    trans = true;

    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    const IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 =
        dim_tensor2 > 1 ? t2_own->sizes().cend()[-2] : t2_own->sizes().back();
    const int64_t p = dim_tensor2 > 1 ? t2_own->sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(
        t2_own->sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0LL));
    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);

    const auto tensor1_expand_size = [&output_shape, n, m1] {
      DimVector ret(output_shape);
      ret.append({n, m1});
      return ret;
    }();
    const auto tensor2_expand_size = [&output_shape, m2, p] {
      DimVector ret(output_shape);
      ret.append({m2, p});
      return ret;
    }();
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                      .reshape({expand_batch_product, n, m1});
    const auto tensor2_expanded = t2_own->expand(tensor2_expand_size)
                                      .reshape({expand_batch_product, m2, p});
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }
    DimVector result_shape({expand_batch_product, n, p});
    Tensor result = output.defined()
        ? output.view(result_shape)
        : at::empty(result_shape, tensor1.options());

    is_fused = get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    xpu::onednn::matmul(
        result, tensor1_expanded, tensor2_expanded, bias, trans, attr);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = at::_unsafe_view(result, output_shape);
    }
    return output;
  }
}

// Matmul_fusion_variants for Meta backend(only query shape)
static Tensor& matmul_fusion_variants_meta(
    Tensor& output,
    const Tensor& tensor1,
    const Tensor& tensor2,
    bool trans,
    xpu::onednn::Attr& attr,
    bool& is_fused,
    Tensor bias = at::Tensor()) {
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();
  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(
      dim_tensor1 != 0 && dim_tensor2 != 0,
      "both arguments to matmul need to be at least 1D, but they are ",
      dim_tensor1,
      "D and ",
      dim_tensor2,
      "D");

  bool should_fold_tensor1 = should_fold(tensor1, dim_tensor2);
  bool should_fold_tensor2 = should_fold(tensor2, dim_tensor1);

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    // case1:
    // original size: [6] x [6] -> []
    is_fused = true;
    output = output.defined() ? output.view({1, 1})
                              : at::empty({1, 1}, tensor1.options());
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    // case2:
    // original sizes: [4, 2] x [2] -> [4]
    // onednn sizes: [4, 2] x [2, 1] -> [4, 1]
    DimVector output_shape({tensor1.size(0)});
    DimVector result_shape({tensor1.size(0), 1});
    output = output.defined() ? output.view(result_shape)
                              : at::empty(result_shape, tensor1.options());
    Tensor t2 = tensor2.view({tensor2.size(0), 1});
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    // case3:
    // original sizes: [2] x [2, 6] -> [6]
    // onednn sizes: [1, 2] x [2, 6] -> [1, 6]
    DimVector output_shape({tensor2.size(1)});
    if (!trans)
      output_shape[0] = tensor2.size(0);
    Tensor t1 = tensor1.unsqueeze(0);
    DimVector result_shape({1, output_shape[0]});
    output = output.defined() ? output.view(result_shape)
                              : at::empty(result_shape, tensor1.options());
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // case4:
    // original sizes: [4, 2] x [2, 6] -> [4, 6]
    // onednn sizes: [4, 2] x [2, 6] -> [4, 6]
    DimVector output_shape({tensor1.size(0), tensor2.size(1)});
    if (!trans)
      output_shape[1] = tensor2.size(0);

    output =
        output.defined() ? output : at::empty(output_shape, tensor1.options());

  } else if (should_fold_tensor1) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2)
    // case5-1:
    // original sizes: [3, 4, 2] x [2, 6] -> [3, 4, 6]
    // onednn sizes: [12, 2] x [2, 6] -> [12, 6]
    // case5-2:
    // original sizes: [3, 4, 2] x [2] -> [3, 4]
    // onednn sizes: [12, 2] x [2, 1] -> [12, 1]
    const auto t1_own = MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2_own = MaybeOwned<Tensor>::borrowed(tensor2);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    if (trans)
      output_shape.push_back(t2.size(1));
    else
      output_shape.push_back(t2.size(0));
    DimVector result_shape({t1.size(0), output_shape[output_shape.size() - 1]});
    output = output.defined() ? output.view(result_shape)
                              : at::empty(result_shape, tensor1.options());
  } else if (should_fold_tensor2) {
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // case6-1:
    // original sizes: [2] x [3, 2, 4] = [3, 4]
    // onednn sizes: [12, 2] x [2, 1] = [12, 1]
    // or
    // original sizes: [2] x [2, 3, 2, 4] = [2, 3, 4]
    // onednn sizes: [24, 2] x [2, 1] = [24, 1]

    // case6-2:
    // original sizes: [6, 2] x [3, 2, 4] = [3, 6, 4]
    // onednn sizes: [12, 2] x [2, 6] = [12, 6]
    // or
    // original sizes: [6, 2] x [2, 3, 2, 4] = [2, 3, 6, 4]
    // onednn sizes: [24, 2] x [2, 6] = [24, 6]

    const auto t1_own = trans
        ? MaybeOwned<Tensor>::owned(tensor2.mT())
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2).mT());
    trans = true;
    const auto t2_own = dim_tensor1 == 2
        ? MaybeOwned<Tensor>::owned(tensor1.t())
        : MaybeOwned<Tensor>::borrowed(tensor1);

    const auto sizes_1 = t1_own->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);
    const auto t1 = t1_own->reshape({folded_dim1, sizes_1.back()});
    const auto t2_is_matrix = t2_own->dim() == 2;
    Tensor t2 = t2_is_matrix ? *t2_own : t2_own->view({t2_own->size(0), 1});
    output_shape.push_back(t2.size(1));
    DimVector result_shape({t1.size(0), t2.size(1)});
    output = output.defined() ? output.view(result_shape)
                              : at::empty(result_shape, tensor1.options());

  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // case7-1:
    // original sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // onednn sizes: [3, 4, 2] x [3, 2, 6] = [3, 4, 6]
    // case7-2:
    // original sizes: [5, 1, 4, 2] x [3, 2, 6] = [5, 3, 4, 6]
    // onednn sizes: [15, 4, 2] x [15, 2, 6] = [15, 4, 6]
    const auto t2_own = trans
        ? MaybeOwned<Tensor>::borrowed(tensor2)
        : MaybeOwned<Tensor>::owned(tensor2.transpose(-1, -2));
    trans = true;

    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    const IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 =
        dim_tensor2 > 1 ? t2_own->sizes().cend()[-2] : t2_own->sizes().back();
    const int64_t p = dim_tensor2 > 1 ? t2_own->sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(
        t2_own->sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0LL));
    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);

    const auto tensor1_expand_size = [&output_shape, n, m1] {
      DimVector ret(output_shape);
      ret.append({n, m1});
      return ret;
    }();
    const auto tensor2_expand_size = [&output_shape, m2, p] {
      DimVector ret(output_shape);
      ret.append({m2, p});
      return ret;
    }();
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                      .reshape({expand_batch_product, n, m1});
    const auto tensor2_expanded = t2_own->expand(tensor2_expand_size)
                                      .reshape({expand_batch_product, m2, p});
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }
    DimVector result_shape({expand_batch_product, n, p});
    output = output.defined() ? output.view(result_shape)
                              : at::empty(result_shape, tensor1.options());
  }
  return output;
}

} // namespace impl

} // namespace xpu
} // namespace at
