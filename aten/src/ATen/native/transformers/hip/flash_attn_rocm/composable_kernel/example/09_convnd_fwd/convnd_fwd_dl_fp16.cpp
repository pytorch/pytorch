// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_dl_common.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using AccDataType = float;
using DsDataType  = ck::Tuple<ck::half_t>;
using OutDataType = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddRelu;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmPadingSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
// clang-format off
using DeviceGroupedConvNDFwdInstance = ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
// ######|        NDim|     InData|     WeiData|   MultpleD|     OutData|     AccData| InLayout| WeiLayout|            MultipleD| OutLayout|           In|           Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
// ######|     Spatial|       Type|        Type|       Type|        Type|        Type|         |          |               Layout|          |  Elementwise|   Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
// ######|            |           |            |           |            |            |         |          |                     |          |    Operation|     Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
// ######|            |           |            |           |            |            |         |          |                     |          |             |              |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
         < NDimSpatial, InDataType, WeiDataType, DsDataType, OutDataType, AccDataType, InLayout, WeiLayout, ck::Tuple<OutLayout>, OutLayout,  InElementOp,  WeiElementOp,  OutElementOp,       ConvSpec,    GemmPadingSpec,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,               5,                 4>;
// clang-format on

#include "run_convnd_fwd_dl_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_dl_example(argc, argv) ? 0 : 1; }
