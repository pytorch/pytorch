// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_dl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using NHWC = ck::tensor_layout::convolution::NHWC;
using KYXC = ck::tensor_layout::convolution::KYXC;
using NHWK = ck::tensor_layout::convolution::NHWK;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;

// Compilation parameters for in[n, hi, wi, c] * wei[k, y, x, c] = out[n, ho, wo, k]
using device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instances = std::tuple<
    // clang-format off
        //#########################|       NDim|     InData|     WeiData|     OutData|     AccData|           In|           Wei|           Out|        Convolution| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //#########################|    Spatial|       Type|        Type|        Type|        Type|  Elementwise|   Elementwise|   Elementwise|            Forward|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //#########################|           |           |            |            |            |    Operation|     Operation|     Operation|     Specialization|      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //#########################|           |           |            |            |            |             |              |              |                   |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
    DeviceConvNdBwdDataNwcKxcNwk_Dl<          2, InDataType, WeiDataType, OutDataType, AccDataType,  InElementOp,  WeiElementOp,  OutElementOp, ConvBwdDataDefault,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<1, 1, 8, 2>,      S<16, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 8, 1>,      S<0, 3, 1, 2>,       S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

using device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_1x1_s1_p0_f16_instances = std::tuple<
    // clang-format off
        //#########################|       NDim|     InData|     WeiData|     OutData|     AccData|           In|           Wei|           Out|                     Convolution| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //#########################|    Spatial|       Type|        Type|        Type|        Type|  Elementwise|   Elementwise|   Elementwise|                         Forward|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //#########################|           |           |            |            |            |    Operation|     Operation|     Operation|                  Specialization|      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //#########################|           |           |            |            |            |             |              |              |                                |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
    DeviceConvNdBwdDataNwcKxcNwk_Dl<          2, InDataType, WeiDataType, OutDataType, AccDataType,  InElementOp,  WeiElementOp,  OutElementOp, ConvBwdDataFilter1x1Stride1Pad0,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<1, 1, 8, 2>,      S<16, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 8, 1>,      S<0, 3, 1, 2>,       S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

void add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
                                                  NHWC,
                                                  KYXC,
                                                  NHWK,
                                                  InDataType,
                                                  WeiDataType,
                                                  OutDataType,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instances{});
    add_device_operation_instances(
        instances, device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_1x1_s1_p0_f16_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
