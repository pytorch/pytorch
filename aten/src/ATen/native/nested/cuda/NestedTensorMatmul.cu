#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef USE_ROCM
#ifndef _WIN32
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#endif
#endif

#include <ATen/NestedTensorImpl.h>

#define BLOCK_DIM 256
#define GRID_DIM_Y 16

namespace at {
namespace native {

#ifndef USE_ROCM
#ifndef _WIN32
namespace {

template <
    typename scalar_t,
    unsigned int kPad,
    typename LayoutA,
    typename LayoutB,
    typename OpClass,
    typename Arch,
    typename ThreadBlockShape,
    typename WarpShape,
    typename InstructionShape>
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

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          Arch,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      Arch,
      ThreadBlockShape,
      WarpShape,
      InstructionShape,
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

template <typename scalar_t>
bool group_gemm_dispatch(
    at::Device device,
    const std::vector<scalar_t*>& aptr,
    const std::vector<scalar_t*>& bptr,
    const std::vector<scalar_t*>& dptr,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {
  return false;
}

template <>
bool group_gemm_dispatch(
    at::Device device,
    const std::vector<float*>& aptr,
    const std::vector<float*>& bptr,
    const std::vector<float*>& dptr,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {

  gemm_grouped_cuda_internal<
      float,
      1,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 8>,
      cutlass::gemm::GemmShape<64, 32, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>>(
      lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
  return true;
}

template <>
bool group_gemm_dispatch(
    at::Device device,
    const std::vector<c10::Half*>& aptr_,
    const std::vector<c10::Half*>& bptr_,
    const std::vector<c10::Half*>& dptr_,
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors) {

  // Check alignment
  bool all_pad_8 = true;
  for (int i = 0; i < ntensors; i++) {
    all_pad_8 = all_pad_8 && (gemm_sizes[i].n() % 8 == 0);
    all_pad_8 = all_pad_8 && (gemm_sizes[i].k() % 8 == 0);

    // Not sure if this is a requirement, on the safe side
    all_pad_8 = all_pad_8 && (lda[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldb[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldd[i] % 8 == 0);
  }

  std::vector<cutlass::half_t*> aptr;
  std::vector<cutlass::half_t*> bptr;
  std::vector<cutlass::half_t*> dptr;
  for (int64_t i = 0; i < ntensors; i++) {
    aptr.push_back(reinterpret_cast<cutlass::half_t*>(aptr_[i]));
    bptr.push_back(reinterpret_cast<cutlass::half_t*>(bptr_[i]));
    dptr.push_back(reinterpret_cast<cutlass::half_t*>(dptr_[i]));
  }
  if (all_pad_8) {
    gemm_grouped_cuda_internal<
        cutlass::half_t,
        8,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>>(
        lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
    return true;
  } else {
    gemm_grouped_cuda_internal<
        cutlass::half_t,
        1,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 8>,
        cutlass::gemm::GemmShape<64, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>>(
        lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
    return true;
  }
  // Did not perform GEMM
  return false;
}

} // namespace

#endif
#endif

Tensor bmm_nested_cuda(const Tensor& self, const Tensor& mat2) {

  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = self.is_nested() ? get_nested_tensor_impl(self) : self.unsafeGetTensorImpl();
  auto mat2_ptr = mat2.is_nested() ? get_nested_tensor_impl(mat2) : mat2.unsafeGetTensorImpl();
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

  // create a contiguous output
  const Tensor& self_sizemat = self.is_nested() ?
      get_nested_tensor_impl(self)->get_nested_sizes() : get_nested_tensor_impl(mat2)->get_nested_sizes();

  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();

  int64_t out_numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = get_size_for_index(self, i), &mat2_shape = get_size_for_index(mat2, i);
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
    out_numel += self_size0 * mat2_size1;
  }

  const Tensor &self_buffer = self.is_nested() ? get_nested_tensor_impl(self)->get_unsafe_storage_as_tensor() : self;
  const Tensor &mat2_buffer = mat2.is_nested() ? get_nested_tensor_impl(mat2)->get_unsafe_storage_as_tensor() : mat2;

  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  auto out_ptr = get_nested_tensor_impl(output);

  const int64_t *out_offsets_ptr = out_ptr->get_storage_offsets().data_ptr<int64_t>();

#ifndef USE_ROCM
#ifndef _WIN32
  bool success = false;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "group_gemm_dispatch", [&] {
        std::vector<scalar_t*> aptr(ntensors);
        std::vector<scalar_t*> bptr(ntensors);
        std::vector<scalar_t*> dptr(ntensors);
        std::vector<int64_t> lda(ntensors);
        std::vector<int64_t> ldb(ntensors);
        std::vector<int64_t> ldd(ntensors);
        std::vector<cutlass::gemm::GemmCoord> gemm_sizes;
        bool all_row_major = true;
        for (int64_t i = 0; i < ntensors; i++) {
          const IntArrayRef& self_shape = get_size_for_index(self, i);
          const IntArrayRef& mat2_shape = get_size_for_index(mat2, i);
          const int64_t &self_size0 = self_shape[0];
          const int64_t &self_size1 = self_shape[1];
          const int64_t &mat2_size0 = mat2_shape[0];
          const int64_t &mat2_size1 = mat2_shape[1];
          gemm_sizes.push_back(
              cutlass::gemm::GemmCoord(self_size0, mat2_size1, self_size1));
          aptr[i] = self_buffer.data_ptr<scalar_t>() + get_offset_for_index(self, i);
          bptr[i] = mat2_buffer.data_ptr<scalar_t>() + get_offset_for_index(mat2, i);
          dptr[i] = out_buffer.data_ptr<scalar_t>() + out_offsets_ptr[i];
          auto self_stride = get_stride_for_index(self, i);
          auto mat2_stride = get_stride_for_index(mat2, i);
          all_row_major = all_row_major && (self_stride[1] == 1);
          all_row_major = all_row_major && (mat2_stride[1] == 1);
          lda[i] = self_stride[0];
          ldb[i] = mat2_stride[0];
          ldd[i] = mat2_size1;
        }
        auto dprops = at::cuda::getCurrentDeviceProperties();
        bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
        if (all_row_major &&
            self.is_contiguous() &&
            mat2.is_contiguous() &&
            is_sm8x) {
          success = group_gemm_dispatch<scalar_t>(
              output.device(),
              aptr,
              bptr,
              dptr,
              lda,
              ldb,
              ldd,
              gemm_sizes,
              ntensors);
        }
      });
  if (success) {
    return output;
  }
#endif
#endif

  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(output_unbind[i],
        self_buffer.as_strided(get_size_for_index(self, i), get_stride_for_index(self, i), get_offset_for_index(self, i)),
        mat2_buffer.as_strided(get_size_for_index(mat2, i), get_stride_for_index(mat2, i), get_offset_for_index(mat2, i)));
  }
  return output;
}

} // namespace native
} // namespace at
