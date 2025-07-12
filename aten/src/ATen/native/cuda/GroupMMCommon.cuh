#pragma once
#include <cutlass/util/packed_stride.hpp>
#include <ATen/native/cuda/ScaledGroupMM.h>

namespace at::cuda::detail {

using Strides = std::array<int64_t, 3>;

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
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
    int64_t a_scale_stride,
    int64_t b_scale_stride,
    bool a_row_major = true,
    bool b_row_major = false) {
  int32_t M = group_count_info.M;
  int32_t N = group_count_info.N;
  int32_t K = group_count_info.K;

  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    delta = offs[tid] - start;
    if (K < 0) {
      CUDA_KERNEL_ASSERT(delta >=0 && "expected ofsets to be greater or equal 0\n");
    }

    // TMA transfers require global memory tensor addresses to be
    // aligned to 16 bytes.
    if (tid < blockDim.x - 1) {
      // Check this requirement for input tensors, in case group
      // addresses are increased along the dynamic dimension.
      if ((K < 0 && a_row_major) ||       // 2D/2D: check along K dimension
          (M < 0 && !a_row_major)) {      // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if ((K < 0 && !b_row_major) ||      // 2D/2D: check along K dimension
          (N < 0 && b_row_major)) {       // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }

      // Check the same requirement for output tensor (that is always
      // contiguous, and in row-major layout).
      if (N < 0) {
        int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
    }
  }
  int64_t lda, ldb, ldoutput;
  if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_3D) {
    // A and output is 2d
    M = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = tid == 0 ? scale_A : scale_A + offs[tid - 1];
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
    }
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
  } else if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_3D_MatrixB_2D) {
    N = delta;
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[1];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = tid == 0 ? scale_B : scale_B + offs[tid - 1];
    }
  } else if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_2D) {
    // A, B is 2d, output is 3d
    K = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * M;
      inputB_scale_ptrs[tid] = scale_B + tid * N;
    }
  } else {
    // A, B, output are 3D
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
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
}

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void prepare_grouped_gemm_data_sm100(
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
    Strides tensor_StrideSFA,
    Strides tensor_StrideSFB,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    bool transpose = false,
    bool a_row_major = true,
    bool b_row_major = false) {

  int32_t M = group_count_info.M;
  int32_t N = group_count_info.N;
  int32_t K = group_count_info.K;

  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    delta = offs[tid] - start;
    if (K < 0) {
      if (!a_row_major && b_row_major) {
        CUDA_KERNEL_ASSERT(delta >=0 && "expected ofsets to be greater or equal 0\n");
      } else  {
        // CUTLASS cannot handle delta=0 here.
        CUDA_KERNEL_ASSERT(delta >0 && "expected ofsets to be greater than 0\n");
      }
    }

    // TMA transfers require global memory tensor addresses to be
    // aligned to 16 bytes.
    if (tid < blockDim.x - 1) {
      // Check this requirement for input tensors, in case group
      // addresses are increased along the dynamic dimension.
      if ((K < 0 && a_row_major) ||       // 2D/2D: check along K dimension
          (M < 0 && !a_row_major)) {      // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if ((K < 0 && !b_row_major) ||      // 2D/2D: check along K dimension
          (N < 0 && b_row_major)) {       // 3D/2D: check along N dimension
        int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }

      // Check the same requirement for output tensor (that is always
      // contiguous, and in row-major layout).
      if (N < 0) {
        int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
        CUDA_KERNEL_ASSERT(
                           delta % align == 0 &&
                           "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
    }
  }
  int64_t lda, ldb, ldoutput;

  if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_3D) {
    // A and output is 2d
    M = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
  } else if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_3D_MatrixB_2D) {
    N = delta;
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb =
        b_row_major ? tensor_StrideB[0] : tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[1];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
  } else if (group_count_info.input_matrix_type == GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_2D) {
    // A, B is 2d, output is 3d
    K = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
  } else {
    // A, B, output are 3D
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
  }

  if (scale_A != nullptr) {
    // We support only scale_A and scale_B are 3D tensors now.
    inputA_scale_ptrs[tid] = scale_A + tid * tensor_StrideSFA[0];
    inputB_scale_ptrs[tid] = scale_B + tid * tensor_StrideSFB[0];
  }
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

  if (transpose) {
    problem_sizes[tid] = ProblemShape(N, M, K);
  } else {
    problem_sizes[tid] = ProblemShape(M, N, K);
  }

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + tid;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + tid;

  if (!transpose) {
    *layout_sfa_ptr =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    *layout_sfb_ptr =
        ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  } else {
    *layout_sfa_ptr =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(N, M, K, 1));
    *layout_sfb_ptr =
        ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(N, M, K, 1));
  }
}
} // namespace at::cuda::detail
