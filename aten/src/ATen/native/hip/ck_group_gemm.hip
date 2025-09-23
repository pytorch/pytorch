#undef __HIP_NO_HALF_CONVERSIONS__
#include <ATen/hip/HIPContext.h>
#include <ATen/Tensor.h>
#include <ATen/TensorAccessor.h>
#include <c10/hip/HIPStream.h>
#include <iostream>
#include <vector>
#include <optional>
#include <type_traits>

// CK headers
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
namespace at {
namespace hip {
namespace detail {

namespace CkTypes {
    using BF16 = ck::bhalf_t;
    using F32 = float;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using ADataType = BF16;
    using BDataType = BF16;
    using EDataType = BF16;
    using AccDataType = F32;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CDEElementOp = PassThrough;
} // namespace CkTypes

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// A re-usable kernel definition template for grouped GEMM.
// This configuration is a known-good setup for BF16 grouped GEMM with a block size of 256.
// The template parameters have been adjusted to fix the compiler errors.
template <typename ALayout, typename BLayout>
using GroupedGemmKernel = ck::tensor_operation::device::DeviceGroupedGemm_Xdl<
    ALayout,
    BLayout,
    ck::Tuple<>,
    ck::tensor_layout::gemm::RowMajor,
    CkTypes::ADataType,
    CkTypes::BDataType,
    CkTypes::AccDataType,
    CkTypes::BF16,
    ck::Tuple<>,
    CkTypes::EDataType,
    CkTypes::AElementOp,
    CkTypes::BElementOp,
    CkTypes::CDEElementOp,
    ck::tensor_operation::device::GemmSpecialization::Default,
    1,   // NumGemmKPrefetchStage
    256, // BlockSize
    256, // MPerBlock
    128, // NPerBlock
    32,  // KPerBlock
    8,   // AK1
    8,   // BK1
    32,  // MPerXdl
    32,  // NPerXdl
    4,   // MXdlPerWave
    2,   // NXdlPerWave
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    1,
    1,
    S<1, 32, 1, 8>,
    8>;

template <typename ALayout, typename BLayout>
void launch_grouped_bgemm_ck_impl(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    at::Tensor& out) {
    using namespace CkTypes;
    using DeviceOp = GroupedGemmKernel<ALayout, BLayout>;

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    // Prepare input groups
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<const void*> p_a_ptrs;
    std::vector<const void*> p_b_ptrs;
    std::vector<void*> p_e_ptrs;

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2, "Only 2D x 2D supported");
    TORCH_CHECK(offs.has_value(), "Offsets tensor required");
    auto offs_accessor = offs->accessor<int, 1>();
    int num_groups = offs_accessor.size(0) - 1;

    // Get the base pointers for the entire tensors
    const char* a_ptr_base = reinterpret_cast<const char*>(mat_a.data_ptr());
    const char* b_ptr_base = reinterpret_cast<const char*>(mat_b.data_ptr());
    char* out_ptr_base = reinterpret_cast<char*>(out.data_ptr());

    // Correctly determine the m dimension for each group
    int m = out.size(0) / num_groups;
    int n = mat_b.size(1);

    // Calculate the size of a single element for pointer arithmetic
    const size_t a_element_size = mat_a.element_size();
    const size_t b_element_size = mat_b.element_size();
    const size_t out_element_size = out.element_size();

    for (int i = 0; i < num_groups; ++i) {
        int k = offs_accessor[i + 1] - offs_accessor[i];
        int start_k = offs_accessor[i];

        // Skip groups with zero length
        if (k == 0) {
            continue;
        }

        const void* group_a_ptr = nullptr;
        const void* group_b_ptr = nullptr;
        void* group_e_ptr = nullptr;

        int lda, ldb, ldc;

        // Correct pointer and stride calculation based on layout
        if (std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value) {
            group_a_ptr = a_ptr_base + start_k * mat_a.stride(1) * a_element_size;
            lda = mat_a.stride(0);
        } else { // ColumnMajor
            group_a_ptr = a_ptr_base + start_k * mat_a.stride(1) * a_element_size;
            lda = mat_a.stride(1);
        }

        if (std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value) {
            group_b_ptr = b_ptr_base + start_k * mat_b.stride(0) * b_element_size;
            ldb = mat_b.stride(0);
        } else { // ColumnMajor
            group_b_ptr = b_ptr_base + start_k * mat_b.stride(1) * b_element_size;
            ldb = mat_b.stride(1);
        }

        ldc = out.stride(0);
        group_e_ptr = out_ptr_base + i * m * out.stride(0) * out_element_size;

        gemm_descs.push_back({static_cast<ck::index_t>(m),
                              static_cast<ck::index_t>(n),
                              static_cast<ck::index_t>(k),
                              static_cast<ck::index_t>(lda),
                              static_cast<ck::index_t>(ldb),
                              static_cast<ck::index_t>(ldc)});
        p_a_ptrs.push_back(group_a_ptr);
        p_b_ptrs.push_back(group_b_ptr);
        p_e_ptrs.push_back(group_e_ptr);
    }

    TORCH_INTERNAL_ASSERT(p_a_ptrs.size() > 0, "No valid groups!");

    static DeviceOp gemm_instance;
    std::vector<std::array<const void*, 0>> p_ds_empty;
    auto argument = gemm_instance.MakeArgument(p_a_ptrs, p_b_ptrs, p_ds_empty, p_e_ptrs,
                                               gemm_descs, PassThrough{}, PassThrough{}, PassThrough{});
    TORCH_INTERNAL_ASSERT(gemm_instance.IsSupportedArgument(argument), "CK GEMM argument unsupported (shape/strides/type config)");
    auto invoker = gemm_instance.MakeInvoker();
    invoker.Run(argument, {stream});
}

void bf16bf16_grouped_mm_ck(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& /*bias*/,
    at::Tensor& out) {
    bool a_row_major = mat_a.stride(1) == 1;
    bool b_row_major = mat_b.stride(1) == 1;
    if (a_row_major && b_row_major) {
        launch_grouped_bgemm_ck_impl<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor>(mat_a, mat_b, offs, out);
    } else if (a_row_major && !b_row_major) {
        launch_grouped_bgemm_ck_impl<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor>(mat_a, mat_b, offs, out);
    } else if (!a_row_major && b_row_major) {
        launch_grouped_bgemm_ck_impl<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor>(mat_a, mat_b, offs, out);
    } else {
        launch_grouped_bgemm_ck_impl<ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::ColumnMajor>(mat_a, mat_b, offs, out);
    }
}
} // namespace detail
} // namespace hip
} // namespace at


