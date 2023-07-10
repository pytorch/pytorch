// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_gnwc_gkxc_gnwk_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using GNWC = ck::tensor_layout::convolution::GNWC;
using GKXC = ck::tensor_layout::convolution::GKXC;
using GNWK = ck::tensor_layout::convolution::GNWK;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

// Compilation parameters for in[n, wi, c] * wei[k, x, c] = out[n, wo, k]
using device_grouped_conv1d_bwd_weight_xdl_c_shuffle_gnwc_gkxc_gnwk_f32_default_instances =
    std::tuple<
        // clang-format off
        //#########################################|     Num| InData| WeiData| OutData| AccData|          In|         Wei|         Out|                      ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer|
        //#########################################|     Dim|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                            Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|
        //#########################################| Spatial|       |        |        |        |   Operation|   Operation|   Operation|                    Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl|
        //#########################################|        |       |        |        |        |            |            |            |                                  |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   256,   256,   128,     4,  4,   32,   32,    4,    2,  S<1, 4, 64, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   256,   128,   256,     4,  4,   32,   32,    2,    4,  S<1, 4, 32, 2>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 64, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   128,   128,   128,     4,  4,   32,   32,    4,    2,  S<1, 4, 32, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   256,   128,   128,     4,  4,   32,   32,    2,    2,  S<1, 4, 32, 2>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   128,   128,    64,     4,  4,   32,   32,    2,    2,  S<1, 4, 32, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 16, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   128,    64,   128,     4,  4,   32,   32,    2,    2,  S<1, 4, 16, 2>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,    64,    64,    64,     4,  4,   32,   32,    2,    2,  S<1, 4, 16, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 16, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   256,   128,    64,     4,  4,   32,   32,    2,    1,  S<1, 4, 32, 2>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 16, 4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              1,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   256,    64,   128,     4,  4,   32,   32,    1,    2,  S<1, 4, 16, 4>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              1,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   128,   128,    32,     4,  4,   32,   32,    2,    1,  S<1, 4, 32, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 8,  4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              1,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,   128,    32,   128,     4,  4,   32,   32,    1,    2,  S<1, 4, 8,  4>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              1,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,    64,    64,    32,     4,  4,   32,   32,    2,    1,  S<1, 4, 16, 1>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 8,  2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 16, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,              ConvBwdWeightDefault,    64,    32,    64,     4,  4,   32,   32,    1,    2,  S<1, 4, 8,  2>,   S<0, 3, 1, 2>,   S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 16, 1, 4>,               4>
        // clang-format on
        >;

using device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_1x1_s1_p0_f32_instances = std::tuple<
    // clang-format off
        //#########################################|     Num| InData| WeiData| OutData| AccData|          In|         Wei|         Out|                      ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer|
        //#########################################|     Dim|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                            Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|
        //#########################################| Spatial|       |        |        |        |   Operation|   Operation|   Operation|                    Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl|
        //#########################################|        |       |        |        |        |            |            |            |                                  |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   256,   256,   128,     4,  4,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   256,   128,   256,     4,  4,   32,   32,    2,    4,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 64, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   128,   128,   128,     4,  4,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   256,   128,   128,     4,  4,   32,   32,    2,    2,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   128,   128,    64,     4,  4,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 16, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   128,    64,   128,     4,  4,   32,   32,    2,    2,  S<1, 4, 16, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,    64,    64,    64,     4,  4,   32,   32,    2,    2,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 16, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   256,   128,    64,     4,  4,   32,   32,    2,    1,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 16, 4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              1,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   256,    64,   128,     4,  4,   32,   32,    1,    2,  S<1, 4, 16, 4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              1,      true,  S<1, 4, 32, 2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   128,   128,    32,     4,  4,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 8,  4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              1,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,   128,    32,   128,     4,  4,   32,   32,    1,    2,  S<1, 4, 8,  4>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              1,      true,  S<1, 4, 32, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 32, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,    64,    64,    32,     4,  4,   32,   32,    2,    1,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              4,      true,  S<1, 4, 8,  2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              2,      true,           1,           1,   S<1, 16, 1, 4>,               4>,
        DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle<       1,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvBwdWeightFilter1x1Stride1Pad0,    64,    32,    64,     4,  4,   32,   32,    1,    2,  S<1, 4, 8,  2>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              4,              2,      true,  S<1, 4, 16, 1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              4,              4,      true,           1,           1,   S<1, 16, 1, 4>,               4>
    // clang-format on
    >;

void add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv1d_bwd_weight_xdl_c_shuffle_gnwc_gkxc_gnwk_f32_default_instances{});
    add_device_operation_instances(
        instances, device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_1x1_s1_p0_f32_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
