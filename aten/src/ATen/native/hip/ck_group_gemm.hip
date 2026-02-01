#undef __HIP_NO_HALF_CONVERSIONS__
#include <ATen/hip/HIPContext.h>
#include <ATen/Tensor.h>
#include <ATen/TensorAccessor.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <iostream>
#include <vector>
#include <optional>
#include <type_traits>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/tuple.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace at {
namespace hip {
namespace detail {

namespace CkTypes {
    using BF16 = ck::bhalf_t;
    using F16 = ck::half_t;
    using F32 = float;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
}

template <typename ALayout, typename BLayout, typename DataType>
using GroupedGemmKernel = ck::tensor_operation::device::DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage<
    ALayout, BLayout, ck::Tuple<>, ck::tensor_layout::gemm::RowMajor,
    DataType, DataType, CkTypes::F32, DataType, ck::Tuple<>, DataType,
    CkTypes::PassThrough, CkTypes::PassThrough, CkTypes::PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2,
    S<1,4,64,1>,  S<0,2,1,3>, S<0,2,1,3>,
    3, 8, 8, 1,
    S<1,4,64,1>,  S<0,2,1,3>, S<0,2,1,3>,
    3, 8, 8, 1,
    1, 1,
    S<1,32,1,8>, 4
>;

template <typename ALayout, typename BLayout, typename DataType>
void launch_grouped_bgemm_ck_impl_dispatch(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    at::Tensor& out)
{
    using DeviceOp = GroupedGemmKernel<ALayout, BLayout, DataType>;
    using PassThrough = CkTypes::PassThrough;

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<const void*> p_a_ptrs, p_b_ptrs;
    std::vector<void*> p_e_ptrs;
    // Note: d_ptrs will be resized after we populate the other vectors

    const int mat_a_dim = mat_a.dim();
    const int mat_b_dim = mat_b.dim();

    const char* a_ptr_base = reinterpret_cast<const char*>(mat_a.data_ptr());
    const char* b_ptr_base = reinterpret_cast<const char*>(mat_b.data_ptr());
    char* out_ptr_base = reinterpret_cast<char*>(out.data_ptr());
    const size_t a_element_size = mat_a.element_size();
    const size_t b_element_size = mat_b.element_size();
    const size_t out_element_size = out.element_size();

    // Synchronize the stream to ensure all prior GPU ops
    // affecting host memory are done.
    AT_CUDA_CHECK(hipStreamSynchronize(c10::hip::getCurrentHIPStream()));

    // for each group, calculate m,n,k,lda,ldb,ldc and A,B,out pointer base addresses.
    if (mat_a_dim == 2 && mat_b_dim == 2) {
        // 2D*2D case requires offset tensor
        auto offs_accessor = offs->accessor<int, 1>();
        int num_groups = offs_accessor.size(0);
        const int M = mat_a.size(0); // number of rows in A
        const int N = mat_b.size(1); // number of columns in B
        const int K = mat_a.size(1); // columns in A == rows in B
        // for 2d*2d input, output is 3d.
        // for each group, A columns (K) are sliced. M and N dimensions are not sliced.
        for (int i = 0; i < num_groups; ++i) {
            int start_k = (i == 0) ? 0 : offs_accessor[i-1];
            int end_k = offs_accessor[i];
            int k = end_k - start_k;

            //K dimension are sliced, hence select stride(1) always.
            //K dimension is always dimension 1, regardless of memory layout (row/column major)
            const void* group_a_ptr = a_ptr_base + start_k * mat_a.stride(1) * a_element_size;
            const void* group_b_ptr;
            int ldb;

            if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major B [K,N]: K values are horizontally adjacent, use stride(1) for K offset
                group_b_ptr = b_ptr_base + start_k * mat_b.stride(1) * b_element_size;
                // Leading dimension = distance between rows = stride(0)
                ldb = mat_b.stride(0);
            } else {
                // Column-major B [K,N]: K values are vertically adjacent, use stride(0) for K offset
                group_b_ptr = b_ptr_base + start_k * mat_b.stride(0) * b_element_size;
                // Leading dimension = distance between columns = stride(1)
                ldb = mat_b.stride(1);
            }

            // Calculate output pointer for group i in 3D tensor [num_groups, M, N]
            // stride(0) = M*N elements between groups, so skip i*stride(0) elements to reach group i
            void* group_e_ptr = out_ptr_base + i * out.stride(0) * out_element_size;
            int lda, ldc;
            if (std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major A [M,K]: leading dimension = distance between rows = stride(0)
                lda = mat_a.stride(0);
            } else {
                // Column-major A [M,K]: leading dimension = distance between columns = stride(1)
                lda = mat_a.stride(1);
            }
            // Output is always row-major in 3D tensor [num_groups, M, N]
            // Leading dimension for each group's [M,N] slice = stride(1) = N
            ldc = out.stride(1);
            size_t output_group_bytes = M * N * out_element_size;
            void* group_e_ptr_end = (char*)group_e_ptr + output_group_bytes;

            gemm_descs.push_back({
                static_cast<ck::index_t>(M),
                static_cast<ck::index_t>(N),
                static_cast<ck::index_t>(k),
                static_cast<ck::index_t>(lda),
                static_cast<ck::index_t>(ldb),
                static_cast<ck::index_t>(ldc),
                {} // --> stride_Ds_
            });
            p_a_ptrs.push_back(group_a_ptr);
            p_b_ptrs.push_back(group_b_ptr);
            p_e_ptrs.push_back(group_e_ptr);
        }
    } else if (mat_a_dim == 2 && mat_b_dim == 3) {
        // 2D*3D case requires offset tensor
        auto offs_accessor = offs->accessor<int, 1>();
        int num_groups = offs_accessor.size(0);

        // 2d*3d input, output is 2d.
        // A: [m * n_groups, k], B: [n_groups, n, k] or [n_groups, k, n], Output: [m * n_groups, n]
        // Offset divides M dimension (rows of A), each group gets different rows of A and different batch of B
        const int K = mat_a.size(1); // columns in A
        // For 2D-3D case: The output determines N (result width)
        const int N = out.size(1); // N is the width of the output tensor

        for (int i = 0; i < num_groups; ++i) {
            int start_m = (i == 0) ? 0 : offs_accessor[i - 1];
            int end_m = offs_accessor[i];
            int m = end_m - start_m;

            // Skip zero-sized groups but continue processing subsequent groups
            if (m <= 0) {
                continue;
            }

            // Select A rows for group i: skip start_m rows
            const void* group_a_ptr;
            int lda;
            if (std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major A [total_m, K]: skip start_m rows, each row is stride(0) elements apart
                group_a_ptr = a_ptr_base + start_m * mat_a.stride(0) * a_element_size;
                lda = mat_a.stride(0); // distance between rows
            } else {
                // Column-major A [total_m, K]: skip start_m elements in the first dimension (stride(0) is between rows)
                group_a_ptr = a_ptr_base + start_m * mat_a.stride(0) * a_element_size;

                // Detect stride pattern for A tensor to determine appropriate lda calculation
                bool a_is_strided_tensor = (mat_a.stride(0) > mat_a.size(0));

                if (a_is_strided_tensor) {
                    // For strided A tensors: stride(0) gives the actual leading dimension
                    lda = mat_a.stride(0);
                } else {
                    // For non-strided A tensors: use the M dimension (total rows)
                    lda = mat_a.size(0); // Total M dimension for column-major layout
                }
            }

            // Select B batch for group i: B[i, :, :]
            const void* group_b_ptr = b_ptr_base + i * mat_b.stride(0) * b_element_size;
            int ldb;

            if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major GEMM: expecting B as [K, N] but we have [N, K], so transpose needed
                ldb = mat_b.stride(2); // Leading dimension for accessing as [K, N]
            } else {
                // Detect stride pattern to determine appropriate ldb calculation
                bool is_strided_tensor = (mat_b.stride(2) > mat_b.size(2));

                if (is_strided_tensor) {
                    // For strided tensors: stride(2) gives the actual leading dimension
                    ldb = mat_b.stride(2);
                } else {
                    // For non-strided tensors: use the N dimension
                    ldb = mat_b.size(1);
                }
            }

            // Output for this group: rows [start_m:end_m, :] in 2D output [total_m, N]
            void* group_e_ptr = out_ptr_base + start_m * out.stride(0) * out_element_size;
            int ldc = out.stride(0); // distance between rows in output (should be N for 2D case)

            gemm_descs.push_back({
                static_cast<ck::index_t>(m),
                static_cast<ck::index_t>(N),
                static_cast<ck::index_t>(K),
                static_cast<ck::index_t>(lda),
                static_cast<ck::index_t>(ldb),
                static_cast<ck::index_t>(ldc),
                {} // --> stride_Ds_
            });
            p_a_ptrs.push_back(group_a_ptr);
            p_b_ptrs.push_back(group_b_ptr);
            p_e_ptrs.push_back(group_e_ptr);
        }
    } else if (mat_a_dim == 3 && mat_b_dim == 3) {
        // 3d*3d input, output is 3d - batched matrix multiplication
        // A: [batch, m, k], B: [batch, k, n] or [batch, n, k] (depending on transpose), Output: [batch, m, n]
        // Each batch is processed as a separate GEMM operation
        const int batch_size = mat_a.size(0);
        const int M = mat_a.size(1); // rows in each A matrix
        const int K = mat_a.size(2); // columns in A == rows in B (or columns if B is transposed)

        // Determine N from B tensor - it could be B.size(1) or B.size(2) depending on layout
        int N;
        if (mat_b.size(1) == K) {
            // B is [batch, k, n] - normal layout
            N = mat_b.size(2);
        } else if (mat_b.size(2) == K) {
            // B is [batch, n, k] - transposed layout
            N = mat_b.size(1);
        } else {
            TORCH_CHECK(false, "CK Group GEMM 3D-3D: B tensor dimensions incompatible with A. A=[",
                       batch_size, ",", M, ",", K, "], B=[", mat_b.size(0), ",", mat_b.size(1), ",", mat_b.size(2), "]");
        }

        for (int i = 0; i < batch_size; ++i) {
            // Select A batch for group i: A[i, :, :]
            const void* group_a_ptr = a_ptr_base + i * mat_a.stride(0) * a_element_size;

            // Select B batch for group i: B[i, :, :]
            const void* group_b_ptr = b_ptr_base + i * mat_b.stride(0) * b_element_size;

            // Select output batch for group i: Output[i, :, :]
            void* group_e_ptr = out_ptr_base + i * out.stride(0) * out_element_size;

            int lda, ldb, ldc;

            if (std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major A: leading dimension = distance between rows = stride(1)
                lda = mat_a.stride(1);
            } else {
                // Column-major A: leading dimension = distance between columns = stride(2)
                lda = mat_a.stride(2);
            }

            if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major B: leading dimension = distance between rows
                if (mat_b.size(1) == K) {
                    // B is [batch, k, n] - normal layout
                    ldb = mat_b.stride(1); // stride between K rows
                } else {
                    // B is [batch, n, k] - transposed layout, treat as [k, n] for GEMM
                    ldb = mat_b.stride(2); // stride between N rows (since we're accessing as [k,n])
                }
            } else {
                // Column-major B: leading dimension = distance between columns
                if (mat_b.size(1) == K) {
                    // B is [batch, k, n] - normal layout
                    ldb = mat_b.stride(2); // stride between N columns
                } else {
                    // B is [batch, n, k] - transposed layout
                    ldb = mat_b.stride(1); // stride between K columns (since we're accessing as [n,k]→[k,n])
                }
            }

            // Output is typically row-major: leading dimension = distance between rows = stride(1)
            ldc = out.stride(1);

            gemm_descs.push_back({
                static_cast<ck::index_t>(M),
                static_cast<ck::index_t>(N),
                static_cast<ck::index_t>(K),
                static_cast<ck::index_t>(lda),
                static_cast<ck::index_t>(ldb),
                static_cast<ck::index_t>(ldc),
                {} // --> stride_Ds_
            });
            p_a_ptrs.push_back(group_a_ptr);
            p_b_ptrs.push_back(group_b_ptr);
            p_e_ptrs.push_back(group_e_ptr);
        }
    } else if (mat_a_dim == 3 && mat_b_dim == 2) {
        // 3D*2D case requires offset tensor
        auto offs_accessor = offs->accessor<int, 1>();
        int num_groups = offs_accessor.size(0);
        // 3d*2d input, output is 3d.
        // A: [n_groups, m, k], B: [k, total_n] (assuming row-major for both)
        // Offset divides N dimension of B, each group gets different slice of B and different batch of A
        const int batch_size = mat_a.size(0); // n_groups
        const int M = mat_a.size(1); // rows in each A matrix
        const int K = mat_a.size(2); // columns in A

        // For row-major A and B case: B should be [K, total_N]
        const int total_N = mat_b.size(1); // B is [K, total_N] for row-major

        for (int i = 0; i < num_groups; ++i) {
            int start_n = (i == 0) ? 0 : offs_accessor[i - 1];
            int end_n = offs_accessor[i];
            int n = end_n - start_n;

            // Skip zero-sized groups but continue processing subsequent groups
            if (n <= 0) {
                continue;
            }

            // Select A batch for group i: A[i, :, :]
            const void* group_a_ptr = a_ptr_base + i * mat_a.stride(0) * a_element_size;

            // Select B slice for group i: B[:, start_n:end_n] (B[K, total_N])
            const void* group_b_ptr;
            int ldb;

            // Check if B is row-major or column-major
            if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
                // Row-major B [K, total_N]: slice columns [start_n:end_n]
                group_b_ptr = b_ptr_base + start_n * mat_b.stride(1) * b_element_size;
                ldb = mat_b.stride(0); // distance between rows (should be total_N)
            } else {
                // Column-major B [K, total_N]: slice columns [start_n:end_n]
                group_b_ptr = b_ptr_base + start_n * mat_b.stride(1) * b_element_size;
                ldb = mat_b.stride(1); // distance between columns (should be K)
            }

            // Select output slice for group i: Output[:, start_n:end_n]
            void* group_e_ptr = out_ptr_base + start_n * out.stride(1) * out_element_size;

            int lda, ldc;

            // Row-major A: leading dimension = distance between rows = stride(1)
            lda = mat_a.stride(1);
            // Output is row-major: leading dimension = distance between rows = stride(0)
            ldc = out.stride(0);

            gemm_descs.push_back({
                static_cast<ck::index_t>(M),
                static_cast<ck::index_t>(n),
                static_cast<ck::index_t>(K),
                static_cast<ck::index_t>(lda),
                static_cast<ck::index_t>(ldb),
                static_cast<ck::index_t>(ldc),
                {} // --> stride_Ds_
            });
            p_a_ptrs.push_back(group_a_ptr);
            p_b_ptrs.push_back(group_b_ptr);
            p_e_ptrs.push_back(group_e_ptr);
        }
    } else {
        TORCH_CHECK(false, "CK Group GEMM: Unsupported dimensions, mat A dim is ", mat_a_dim, ", mat B dim is ", mat_b_dim);
    }

    TORCH_INTERNAL_ASSERT(p_a_ptrs.size() > 0, "CK Group GEMM: No valid groups");

    // Initialize d_ptrs with the correct size
    std::vector<std::array<const void*, 0>> d_ptrs(p_a_ptrs.size());

    static DeviceOp gemm_instance;
    auto argument = gemm_instance.MakeArgument(
        p_a_ptrs, p_b_ptrs, d_ptrs, p_e_ptrs,
        gemm_descs, PassThrough{}, PassThrough{}, PassThrough{}
    );
    TORCH_INTERNAL_ASSERT(gemm_instance.IsSupportedArgument(argument),
        "CK Group GEMM: argument unsupported (shape/strides/type config)");
    size_t arg_buf_size = gemm_instance.GetDeviceKernelArgSize(&argument);
    size_t ws_size = gemm_instance.GetWorkSpaceSize(&argument);

    void* gemm_arg_buf = c10::hip::HIPCachingAllocator::raw_alloc(arg_buf_size);
    void* ws_buf = c10::hip::HIPCachingAllocator::raw_alloc(ws_size);

    gemm_instance.SetDeviceKernelArgs(&argument, gemm_arg_buf);
    gemm_instance.SetWorkSpacePointer(&argument, ws_buf);

    auto invoker = gemm_instance.MakeInvoker();
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    invoker.Run(argument, {stream});
    c10::hip::HIPCachingAllocator::raw_delete(gemm_arg_buf);
    c10::hip::HIPCachingAllocator::raw_delete(ws_buf);
}

void group_gemm_ck(
    const at::Tensor& input_a,
    const at::Tensor& input_b_colmajor,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& /*bias*/,
    at::Tensor& out)
{
    // Detect if input_a is row-major based on stride pattern
    bool a_row_major = (input_a.dim() == 3) ? (input_a.stride(2) == 1) : (input_a.stride(1) == 1);
    bool b_col_major = (input_b_colmajor.dim() == 3) ? (input_b_colmajor.stride(1) == 1) : (input_b_colmajor.stride(0) == 1);
    // Ensure tensor A is row-major and contiguous if not already
    at::Tensor mat_a = input_a;
    if (!a_row_major) {
        // If A is not row-major, make it contiguous (row-major)
        mat_a = input_a.contiguous();
    }
    // Force tensor B to be column-major using double transpose trick
    // This guarantees stride(0) == 1 and stride(1) == K for [K, N] shape
    at::Tensor mat_b = input_b_colmajor;
    if (!b_col_major) {
        mat_b = input_b_colmajor.transpose(-2, -1).contiguous().transpose(-2, -1);
    }

    // For 3D tensors, check the last dimension stride for row-major detection
    a_row_major = (mat_a.dim() == 3) ? (mat_a.stride(2) == 1) : (mat_a.stride(1) == 1);
    bool b_row_major = (mat_b.dim() == 3) ? (mat_b.stride(2) == 1) : (mat_b.stride(1) == 1);

    if (mat_a.dtype() == at::kBFloat16) {
        // bf16 path
        if (a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::BF16>(mat_a, mat_b, offs, out);
        } else if (a_row_major && !b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::BF16>(mat_a, mat_b, offs, out);
        } else if (!a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::BF16>(mat_a, mat_b, offs, out);
        } else {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::BF16>(mat_a, mat_b, offs, out);
        }
    } else if (mat_a.dtype() == at::kHalf) {
        // fp16 path
        if (a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::F16>(mat_a, mat_b, offs, out);
        } else if (a_row_major && !b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::F16>(mat_a, mat_b, offs, out);
        } else if (!a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::F16>(mat_a, mat_b, offs, out);
        } else {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::F16>(mat_a, mat_b, offs, out);
        }
    } else if (mat_a.dtype() == at::kFloat) {
        // fp32 path
        if (a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::F32>(mat_a, mat_b, offs, out);
        } else if (a_row_major && !b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::F32>(mat_a, mat_b, offs, out);
        } else if (!a_row_major && b_row_major) {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor, CkTypes::F32>(mat_a, mat_b, offs, out);
        } else {
            launch_grouped_bgemm_ck_impl_dispatch<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::ColumnMajor, CkTypes::F32>(mat_a, mat_b, offs, out);
        }
    } else {
        TORCH_CHECK(false, "CK Group GEMM: Unsupported mat_a dtype");
    }

}

} // namespace detail
} // namespace hip
} // namespace at
