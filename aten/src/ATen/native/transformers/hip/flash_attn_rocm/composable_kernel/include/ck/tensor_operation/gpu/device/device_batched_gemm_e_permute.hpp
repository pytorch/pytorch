#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct BatchedGemmEPermuteDesc
{
    ck::index_t G0_, G1_, M_, N_;
    ck::index_t stride_G0_, stride_G1_, stride_M_, stride_N_;
};

template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceBatchedGemmEPermute : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_A,
                        index_t stride_B,
                        index_t batch_stride_A,
                        index_t batch_stride_B,
                        BatchedGemmEPermuteDesc batched_gemm_e_permute_desc,
                        index_t BatchCount,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
