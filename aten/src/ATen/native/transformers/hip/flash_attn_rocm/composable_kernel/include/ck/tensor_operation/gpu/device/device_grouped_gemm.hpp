#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmDesc
{
    ck::index_t M_, N_, K_;
    ck::index_t stride_A_, stride_B_, stride_C_;

    std::vector<ck::index_t> stride_Ds_;
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsisiten NumDTensor");

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<std::array<const void*, NumDTensor>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmDesc>& gemm_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
