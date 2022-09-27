#include <c10/util/Exception.h>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded.h>
#endif

#include <ATen/native/nested/NestedTensorUtils.h>

#include <ATen/cuda/CUDAContext.h>

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
void gemm_grouped_cuda_internal(
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    const std::vector<scalar_t*>& aptr,
    const std::vector<scalar_t*>& bptr,
    const std::vector<scalar_t*>& dptr,
    const std::vector<cutlass::gemm::GemmCoord>& gemm_sizes,
    const int problem_count,
    at::Device& device) {
  using Element = scalar_t;
  using ElementAcc = float;
  using OpClass = cutlass::arch::OpClassSimt;

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          cutlass::arch::Sm80,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      GemmConfiguration::kAlignmentA,
      Element,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      GemmConfiguration::kAlignmentB,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      cutlass::arch::Sm80,
      typename GemmConfiguration::ThreadblockShape,
      typename GemmConfiguration::WarpShape,
      typename GemmConfiguration::InstructionShape,
      typename GemmConfiguration::EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      GemmConfiguration::kStages>::GemmKernel;

  using GemmGrouped = typename cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(/*alpha*/ 1, /*beta*/ 0);

  const int64_t gemm_coord_size =
      problem_count * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
  // Number of gmm args not including *problem_sizes
  at::Tensor gmm_args = at::empty(
      {problem_count * 6 + gemm_coord_size},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  // Obtain pointers for each argument (on host)
  int64_t* lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  int64_t* ldb_data = lda_data + problem_count;
  int64_t* ldd_data = lda_data + 2 * problem_count;
  int64_t* ptr_a_data = lda_data + 3 * problem_count;
  int64_t* ptr_b_data = lda_data + 4 * problem_count;
  int64_t* ptr_d_data = lda_data + 5 * problem_count;
  cutlass::gemm::GemmCoord* problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Set arguments into gmm_args from input args
  for (int i = 0; i < problem_count; ++i) {
    problem_sizes_data[i] = gemm_sizes[i];
    lda_data[i] = lda[i];
    ldb_data[i] = ldb[i];
    ldd_data[i] = ldd[i];
    ptr_a_data[i] = reinterpret_cast<int64_t>(aptr[i]);
    ptr_b_data[i] = reinterpret_cast<int64_t>(bptr[i]);
    ptr_d_data[i] = reinterpret_cast<int64_t>(dptr[i]);
  }
  const int threadblock_count =
      GemmGrouped::sufficient(problem_sizes_data, problem_count);

  // Transfer arguments to GPU
  gmm_args = gmm_args.to(device, true);

  // Obtain pointers for each of arguments (on GPU)
  lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  ldb_data = lda_data + problem_count;
  ldd_data = lda_data + 2 * problem_count;
  ptr_a_data = lda_data + 3 * problem_count;
  ptr_b_data = lda_data + 4 * problem_count;
  ptr_d_data = lda_data + 5 * problem_count;
  problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Create GemmGrouped::Arguments using the arguments prepared above
  typename GemmGrouped::Arguments args(
      problem_sizes_data,
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<Element**>(ptr_a_data),
      reinterpret_cast<Element**>(ptr_b_data),
      reinterpret_cast<Element**>(ptr_d_data),
      reinterpret_cast<Element**>(ptr_d_data),
      lda_data,
      ldb_data,
      ldd_data,
      ldd_data);

  GemmGrouped gemm;
  cutlass::Status status =
      gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

  // Run CUTLASS group GEMM
  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

Tensor bmm_nested_cuda(const Tensor& self, const Tensor& mat2) {
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = get_nested_tensor_impl(self);
  auto mat2_ptr = get_nested_tensor_impl(mat2);
  TORCH_CHECK(self_ptr->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_ptr->dim() == 3, "batch2 must be a 3D tensor");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");
  const Tensor &self_buffer = self_ptr->get_buffer(),
               &mat2_buffer = mat2_ptr->get_buffer();
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr),
                           mat2_sizes = NestedTensor_get_sizes(mat2_ptr),
                           self_strides = NestedTensor_get_strides(self_ptr),
                           mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>&self_offsets = self_ptr->get_offsets(),
        &mat2_offsets = mat2_ptr->get_offsets();

  // create a contiguous output
  int64_t out_numel = 0;
  int64_t a_numel = 0;
  int64_t b_numel = 0;
  const Tensor& self_sizemat = self_ptr->get_nested_size_tensor();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  std::vector<int64_t> output_offsets;
  std::vector<int64_t> a_offsets;
  std::vector<int64_t> b_offsets;
  std::vector<int64_t> lda;
  std::vector<int64_t> ldb;
  std::vector<int64_t> ldd;
  std::vector<cutlass::gemm::GemmCoord> gemm_sizes;
  bool all_row_major = true;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = self_sizes[i], &mat2_shape = mat2_sizes[i];
    const int64_t &self_size0 = self_shape[0], &self_size1 = self_shape[1],
                  &mat2_size0 = mat2_shape[0], &mat2_size1 = mat2_shape[1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        i,
        "-th nested matrices in batch cannot be multiplied (",
        self_size0,
        "x",
        self_size1,
        " and ",
        mat2_size0,
        "x",
        mat2_size1,
        ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    output_offsets.push_back(out_numel);
    out_numel += self_size0 * mat2_size1;
    gemm_sizes.push_back(
        cutlass::gemm::GemmCoord(self_size0, mat2_size1, self_size1));
    lda.push_back(self_strides[i][0]);
    ldb.push_back(mat2_strides[i][0]);
    ldd.push_back(mat2_size1);
    a_offsets.push_back(a_numel);
    b_offsets.push_back(b_numel);
    a_numel += self_size0 * self_strides[i][0];
    b_numel += mat2_size0 * mat2_strides[i][0];
    all_row_major = all_row_major && (self_strides[i][1] == 1);
    all_row_major = all_row_major && (mat2_strides[i][1] == 1);
  }
  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  at::Device device = output.device();

  if (all_row_major && (self.dtype() == at::kFloat)) {
    std::vector<float*> aptr;
    std::vector<float*> bptr;
    std::vector<float*> dptr;
    for (int64_t i = 0; i < ntensors; i++) {
      aptr.push_back(self_buffer.data_ptr<float>() + a_offsets[i]);
      bptr.push_back(mat2_buffer.data_ptr<float>() + b_offsets[i]);
      dptr.push_back(out_buffer.data_ptr<float>() + output_offsets[i]);
    }
    gemm_grouped_cuda_internal<float>(
        lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
  } else if (all_row_major && (self.dtype() == at::kHalf)) {
    std::vector<c10::Half*> aptr;
    std::vector<c10::Half*> bptr;
    std::vector<c10::Half*> dptr;
    for (int64_t i = 0; i < ntensors; i++) {
      aptr.push_back(self_buffer.data_ptr<c10::Half>() + a_offsets[i]);
      bptr.push_back(mat2_buffer.data_ptr<c10::Half>() + b_offsets[i]);
      dptr.push_back(out_buffer.data_ptr<c10::Half>() + output_offsets[i]);
    }
    gemm_grouped_cuda_internal<c10::Half>(
        lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
  } else {
    std::vector<Tensor> output_unbind = output.unbind();
    for (int64_t i = 0; i < ntensors; i++) {
      at::mm_out(
          output_unbind[i],
          self_buffer.as_strided(
              self_sizes[i], self_strides[i], self_offsets[i]),
          mat2_buffer.as_strided(
              mat2_sizes[i], mat2_strides[i], mat2_offsets[i]));
    }
  }
  return output;
}

} // namespace native
} // namespace at
