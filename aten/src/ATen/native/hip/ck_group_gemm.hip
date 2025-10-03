#undef __HIP_NO_HALF_CONVERSIONS__
#include <ATen/hip/HIPContext.h>
#include <ATen/Tensor.h>
#include <ATen/TensorAccessor.h>
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
    std::vector<std::array<const void*, 0>> d_ptrs(p_a_ptrs.size());

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2, "Only 2D x 2D supported");
    TORCH_CHECK(offs.has_value(), "Offsets tensor required");

    auto offs_accessor = offs->accessor<int, 1>();
    int num_groups = offs_accessor.size(0);
    const char* a_ptr_base = reinterpret_cast<const char*>(mat_a.data_ptr());
    const char* b_ptr_base = reinterpret_cast<const char*>(mat_b.data_ptr());
    char* out_ptr_base = reinterpret_cast<char*>(out.data_ptr());
    int m = mat_a.size(0); // number of rows in A
    int n = mat_b.size(1); // number of columns in B
    const size_t a_element_size = mat_a.element_size();
    const size_t b_element_size = mat_b.element_size();
    const size_t out_element_size = out.element_size();

    for (int i = 0; i < num_groups; ++i) {
        int start_k = (i == 0) ? 0 : offs_accessor[i-1];
        int end_k = offs_accessor[i];
        int k = end_k - start_k;

        const void* group_a_ptr = a_ptr_base + start_k * mat_a.stride(1) * a_element_size;
        const void* group_b_ptr;
        int ldb;

        if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
            group_b_ptr = b_ptr_base + start_k * mat_b.stride(1) * b_element_size;
            ldb = mat_b.stride(0);
        } else {
            group_b_ptr = b_ptr_base + start_k * mat_b.stride(0) * b_element_size;
            ldb = mat_b.stride(1);
        }

        void* group_e_ptr = out_ptr_base + i * out.stride(0) * out_element_size;
        int lda = mat_a.stride(0), ldc = out.stride(1);
        size_t output_group_bytes = m * n * out_element_size;
        void* group_e_ptr_end = (char*)group_e_ptr + output_group_bytes;

        gemm_descs.push_back({
            static_cast<ck::index_t>(m),
            static_cast<ck::index_t>(n),
            static_cast<ck::index_t>(k),
            static_cast<ck::index_t>(lda),
            static_cast<ck::index_t>(ldb),
            static_cast<ck::index_t>(ldc)
        });
        p_a_ptrs.push_back(group_a_ptr);
        p_b_ptrs.push_back(group_b_ptr);
        p_e_ptrs.push_back(group_e_ptr);
    }
    TORCH_INTERNAL_ASSERT(p_a_ptrs.size() > 0, "No valid groups!");
    static DeviceOp gemm_instance;
    auto argument = gemm_instance.MakeArgument(
        p_a_ptrs, p_b_ptrs, d_ptrs, p_e_ptrs,
        gemm_descs, PassThrough{}, PassThrough{}, PassThrough{}
    );
    TORCH_INTERNAL_ASSERT(gemm_instance.IsSupportedArgument(argument),
        "CK GEMM argument unsupported (shape/strides/type config)");
    size_t arg_buf_size = gemm_instance.GetDeviceKernelArgSize(&argument);
    size_t ws_size = gemm_instance.GetWorkSpaceSize(&argument);

    void* gemm_arg_buf = nullptr;
    void* ws_buf = nullptr;

    hipMalloc(&gemm_arg_buf, arg_buf_size);
    hipMalloc(&ws_buf, ws_size);

    gemm_instance.SetDeviceKernelArgs(&argument, gemm_arg_buf);
    gemm_instance.SetWorkSpacePointer(&argument, ws_buf);

    auto invoker = gemm_instance.MakeInvoker();
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    invoker.Run(argument, {stream});
    hipFree(gemm_arg_buf);
    hipFree(ws_buf);
}

void group_gemm_ck(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& /*bias*/,
    at::Tensor& out)
{
    bool a_row_major = mat_a.stride(1) == 1;
    bool b_row_major = mat_b.stride(1) == 1;

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
        TORCH_CHECK(false, "Unsupported mat_a dtype for grouped MM");
    }
}

} // namespace detail
} // namespace hip
} // namespace at

