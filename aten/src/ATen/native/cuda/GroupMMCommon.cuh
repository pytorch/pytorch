#pragma once
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <cutlass/util/packed_stride.hpp>

namespace at::cuda::detail {

using Strides = std::array<int64_t, 3>;

struct Sm90ScalingFormat {
  int64_t a_scale_stride = 0;
  int64_t b_scale_stride = 0;

  template <bool is_a, bool is_2D = true>
  __device__ __forceinline__ size_t
  get_input_scale_ptr_offset(int32_t tid, const int32_t* offs) {
    auto stride = is_a ? a_scale_stride : b_scale_stride;
    if constexpr (is_2D) {
      return tid == 0 ? 0 : offs[tid - 1];
    } else {
      return tid * stride;
    }
  }

  __device__ __forceinline__ void setup_layout(
      int32_t tid,
      const int32_t* offs,
      int32_t M,
      int32_t N,
      int32_t K) {}
};

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
struct Sm100ScalingFormat {
  Strides tensor_StrideSFA = {};
  Strides tensor_StrideSFB = {};
  LayoutSFA* layout_sfa_ptr = nullptr;
  LayoutSFB* layout_sfb_ptr = nullptr;

  template <bool is_a, bool use_offset = true>
  __device__ __forceinline__ size_t
  get_input_scale_ptr_offset(int32_t tid, const int32_t* offs) {
    auto stride = is_a ? tensor_StrideSFA[0] : tensor_StrideSFB[0];
    if constexpr (use_offset) {
      return tid == 0 ? 0 : offs[tid - 1] * stride;
    } else {
      return tid * stride;
    }
  }

  __device__ __forceinline__ void setup_layout(
      int32_t tid,
      const int32_t* offs,
      int32_t M,
      int32_t N,
      int32_t K) {
    layout_sfa_ptr[tid] =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_sfb_ptr[tid] =
        ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  }
};

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput,
    typename ScalingFormat>
__global__ void prepare_grouped_gemm_data(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeScale* scale_A,
    DtypeScale* scale_B,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    DtypeScale** inputA_scale_ptrs,
    DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    // Strides for cutlass, cute::Stride
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    GroupCountInfo group_count_info,
    // Original strides of the input tensors
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOutput,
    Strides tensor_ShapeA,
    Strides tensor_ShapeB,
    ScalingFormat scaling_format,
    bool a_row_major = true,
    bool b_row_major = false) {

  // The M, N, K may need to be recalculated from the offs tensor
  int32_t M = group_count_info.M;
  int32_t N = group_count_info.N;
  int32_t K = group_count_info.K;

  auto input_type = group_count_info.input_matrix_type;
  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  int32_t offset = 0;

  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    offset = offs[tid];
    delta = offset - start;
    if (input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_2D) {
      CUDA_KERNEL_ASSERT(
          delta >= 0 && "expected ofsets to be greater or equal 0\n");
    }

    // TMA transfers require global memory tensor addresses to be
    // aligned to 16 bytes.
    if (tid < blockDim.x - 1) {
      // Check this requirement for input tensors, in case group
      // addresses are increased along the dynamic dimension.
      if ((input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_2D &&
           a_row_major) || // 2D/2D: check along K dimension
          (input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_3D &&
           !a_row_major)) { // 3D/2D: check along M dimension
        int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
        CUDA_KERNEL_ASSERT(
            delta % align == 0 &&
            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if ((input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_2D &&
           !b_row_major) || // 2D/2D: check along K dimension
          (input_type == GroupMMInputMatrixType::MatrixA_3D_MatrixB_2D &&
           b_row_major)) { // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
        CUDA_KERNEL_ASSERT(
            delta % align == 0 &&
            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }

      // Check the same requirement for output tensor (that is always
      // contiguous, and in row-major layout).
      if (input_type == GroupMMInputMatrixType::MatrixA_3D_MatrixB_2D) {
        int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
        CUDA_KERNEL_ASSERT(
            delta % align == 0 &&
            "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
    }
  }

  int64_t lda{}, ldb{}, ldoutput{};

  if (input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_3D) {
    // A and output is 2d
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeA[0] && "expected offset to be less than tensor size\n");
    M = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A +
          scaling_format.template get_input_scale_ptr_offset<true, true>(tid, offs);
      inputB_scale_ptrs[tid] = scale_B +
          scaling_format.template get_input_scale_ptr_offset<false, false>(tid, offs);
    }
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
  } else if (input_type == GroupMMInputMatrixType::MatrixA_3D_MatrixB_2D) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeB[0] && "expected offset to be less than tensor size\n");
    N = delta;
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[1];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A +
          scaling_format.template get_input_scale_ptr_offset<true, false>(tid, offs);
      inputB_scale_ptrs[tid] = scale_B +
          scaling_format.template get_input_scale_ptr_offset<false, true>(tid, offs);
    }
  } else if (input_type == GroupMMInputMatrixType::MatrixA_2D_MatrixB_2D) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeA[1] && offset <= tensor_ShapeB[0] && "expected offset to be less than tensor size\n");
    // A, B is 2d, output is 3d
    K = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      // for scale 2D/2D, we shift the pointer by M/N
      inputA_scale_ptrs[tid] = scale_A +
          scaling_format.template get_input_scale_ptr_offset<true, false>(tid, offs);
      inputB_scale_ptrs[tid] = scale_B +
          scaling_format.template get_input_scale_ptr_offset<false, false>(tid, offs);
    }
  } else {
    // A, B, output is 3D
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A +
          scaling_format.template get_input_scale_ptr_offset<true, false>(tid, offs);
      inputB_scale_ptrs[tid] = scale_B +
          scaling_format.template get_input_scale_ptr_offset<false, false>(tid, offs);
    }
  }
  problem_sizes[tid] = ProblemShape(M, N, K);

  // make_cute_packed_stride only replaces one of the stride elements with
  // one the provided values in the shape arguments
  // the indices of the src/dst depend on whether A/B are row-major
  // so constructing shape argument with two similar lda values
  // while it looks non-sensical (and it is a nonsensical shape)
  // is fine for these stride construction purposes - the one that will be used
  // for replacement is correct, the other one is ignored, and we don't have to
  // branch on whether A/B are row-major
  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] =
      cutlass::make_cute_packed_stride(StrideOutput{}, {M, ldoutput, 1});

  scaling_format.setup_layout(tid, offs, M, N, K);
}

} // namespace at::cuda::detail
