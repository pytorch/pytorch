// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include <vector>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "profiler/profile_batched_gemm_softmax_gemm_permute_impl.hpp"

using ck::tensor_operation::device::GemmSpecialization;
using ck::tensor_operation::device::MaskingSpecialization;
using ck::tensor_operation::device::TensorSpecialization;

template <ck::index_t N>
using I = ck::Number<N>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
struct TestBatchedGemmMaskingScaleSoftmaxGemmPermute : public ::testing::Test
{
    using NumDimGType      = std::tuple_element_t<0, Tuple>;
    using NumDimMType      = std::tuple_element_t<1, Tuple>;
    using NumDimNType      = std::tuple_element_t<2, Tuple>;
    using NumDimKType      = std::tuple_element_t<3, Tuple>;
    using NumDimOType      = std::tuple_element_t<4, Tuple>;
    using ADataType        = std::tuple_element_t<5, Tuple>;
    using B0DataType       = std::tuple_element_t<6, Tuple>;
    using B1DataType       = std::tuple_element_t<7, Tuple>;
    using CDataType        = std::tuple_element_t<8, Tuple>;
    using Acc0BiasDataType = std::tuple_element_t<9, Tuple>;
    using Acc1BiasDataType = std::tuple_element_t<10, Tuple>;
    using MaskingType      = std::tuple_element_t<11, Tuple>;

    std::vector<std::vector<int>> lengths_ = {
        {256, 256, 64, 64, 6, 4},
        {256, 256, 128, 128, 4, 6},
        {512, 512, 64, 64, 3, 2},
        {512, 512, 128, 128, 2, 3},
        {1024, 1024, 64, 64, 3, 1},
        {1024, 1024, 128, 128, 1, 1},
    };
    bool bench_  = false;
    bool verify_ = true;

    void RunSingle(int M, int N, int K, int O, int G0, int G1)
    {
        bool pass =
            ck::profiler::profile_batched_gemm_softmax_gemm_permute_impl<NumDimGType::value,
                                                                         NumDimMType::value,
                                                                         NumDimNType::value,
                                                                         NumDimKType::value,
                                                                         NumDimOType::value,
                                                                         ADataType,
                                                                         B0DataType,
                                                                         B1DataType,
                                                                         CDataType,
                                                                         ck::Tuple<>,
                                                                         ck::Tuple<>,
                                                                         MaskingType::value>(
                verify_, 2, false, bench_, M, N, K, O, G0, G1);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M  = lengths[0];
            int N  = lengths[1];
            int K  = lengths[2];
            int O  = lengths[3];
            int G0 = lengths[4];
            int G1 = lengths[5];

            this->RunSingle(M, N, K, O, G0, G1);
        }
    }
};

template <GemmSpecialization GemmSpec>
struct DeviceInstanceWrapper_G2M1N1K1O1_TNTT_FP16_M128_N128_K32_O128
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using Scale       = ck::tensor_operation::element_wise::Scale;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    using ADataType        = F16;
    using B0DataType       = F16;
    using B1DataType       = F16;
    using AccDataType      = float;
    using CShuffleDataType = F16;
    using CDataType        = F16;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    // static constexpr auto GemmSpec = std::tuple_element_t<0, Tuple>::value;

    using DeviceGemmGemmInstance =
        ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
            2,
            1,
            1,
            1,
            1,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            ck::Tuple<>,
            ck::Tuple<>,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            TensorSpecialization::Default, // ATensorSpec
            TensorSpecialization::Default, // B0TensorSpec
            TensorSpecialization::Default, // B1TensorSpec
            TensorSpecialization::Default, // CTensorSpec
            1,
            256,
            128,         // MPerBlock
            128,         // NPerBlock
            32,          // KPerBlock
            128,         // Gemm1NPerBlock
            32,          // Gemm1KPerBlock
            8,           // AK1
            8,           // BK1
            2,           // B1K1
            32,          // MPerXDL
            32,          // NPerXDL
            1,           // MXdlPerWave
            4,           // NXdlPerWave
            4,           // Gemm1NXdlPerWave
            S<4, 64, 1>, // ABlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>, // BBlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<8, 32, 1>, // B1BlockTransfer
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            4,
            2,
            false,
            1,              // CShuffleMXdlPerWavePerShuffle
            2,              // CShuffleNXdlPerWavePerShuffle
            S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
            MaskingSpecialization::MaskOutUpperTriangle>; // MaskOutUpperTriangle

    bool IsSupported(int M, int N, int K, int O)
    {
        const int G0 = 1, G1 = 1;

        // A layout [G0, M, G1, K]
        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides{M * G1 * K, K, G1 * K, 1};

        // B0 layout [G0, N, G1, K]
        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides{N * G1 * K, K, G1 * K, 1};

        // B1 layout [G0, N, G1, O]
        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides{N * G1 * O, O, 1, G1 * O};

        // C layout [G0, M, G1, O]
        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};

        auto gemm     = DeviceGemmGemmInstance{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<ADataType*>(nullptr),
                                          static_cast<B0DataType*>(nullptr),
                                          static_cast<B1DataType*>(nullptr),
                                          static_cast<CDataType*>(nullptr),
                                          {}, // p_acc0_biases
                                          {}, // p_acc1_biases
                                          a_gs_ms_ks_lengths,
                                          a_gs_ms_ks_strides,
                                          b0_gs_ns_ks_lengths,
                                          b0_gs_ns_ks_strides,
                                          b1_gs_os_ns_lengths,
                                          b1_gs_os_ns_strides,
                                          c_gs_ms_os_lengths,
                                          c_gs_ms_os_strides,
                                          {},             // acc0_biases_gs_ms_ns_lengths
                                          {},             // acc0_biases_gs_ms_ns_strides
                                          {},             // acc1_biases_gs_ms_os_lengths
                                          {},             // acc1_biases_gs_ms_os_strides
                                          PassThrough{},  // a_element_op
                                          PassThrough{},  // b0_element_op
                                          Scale{1.f},     // acc0_element_op
                                          PassThrough{},  // b1_element_op
                                          PassThrough{}); // c_element_op

        return gemm.IsSupportedArgument(argument);
    }
};

template <GemmSpecialization GemmSpec>
struct DeviceInstanceWrapper_G2M1N1K1O1_TNTT_BF16_M128_N128_K32_O128
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using Scale       = ck::tensor_operation::element_wise::Scale;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    using ADataType        = BF16;
    using B0DataType       = BF16;
    using B1DataType       = BF16;
    using AccDataType      = float;
    using CShuffleDataType = BF16;
    using CDataType        = BF16;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    // static constexpr auto GemmSpec = std::tuple_element_t<0, Tuple>::value;

    using DeviceGemmGemmInstance =
        ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
            2,
            1,
            1,
            1,
            1,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            ck::Tuple<>,
            ck::Tuple<>,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            TensorSpecialization::Default, // ATensorSpec
            TensorSpecialization::Default, // B0TensorSpec
            TensorSpecialization::Default, // B1TensorSpec
            TensorSpecialization::Default, // CTensorSpec
            1,
            256,
            128,         // MPerBlock
            128,         // NPerBlock
            32,          // KPerBlock
            128,         // Gemm1NPerBlock
            32,          // Gemm1KPerBlock
            8,           // AK1
            8,           // BK1
            2,           // B1K1
            32,          // MPerXDL
            32,          // NPerXDL
            1,           // MXdlPerWave
            4,           // NXdlPerWave
            4,           // Gemm1NXdlPerWave
            S<4, 64, 1>, // ABlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>, // BBlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<8, 32, 1>, // B1BlockTransfer
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            4,
            2,
            false,
            1,              // CShuffleMXdlPerWavePerShuffle
            2,              // CShuffleNXdlPerWavePerShuffle
            S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
            MaskingSpecialization::MaskOutUpperTriangle>; // MaskOutUpperTriangle

    bool IsSupported(int M, int N, int K, int O)
    {
        const int G0 = 1, G1 = 1;

        // A layout [G0, M, G1, K]
        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides{M * G1 * K, K, G1 * K, 1};

        // B0 layout [G0, N, G1, K]
        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides{N * G1 * K, K, G1 * K, 1};

        // B1 layout [G0, N, G1, O]
        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides{N * G1 * O, O, 1, G1 * O};

        // C layout [G0, M, G1, O]
        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};

        auto gemm     = DeviceGemmGemmInstance{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<ADataType*>(nullptr),
                                          static_cast<B0DataType*>(nullptr),
                                          static_cast<B1DataType*>(nullptr),
                                          static_cast<CDataType*>(nullptr),
                                          {}, // p_acc0_biases
                                          {}, // p_acc1_biases
                                          a_gs_ms_ks_lengths,
                                          a_gs_ms_ks_strides,
                                          b0_gs_ns_ks_lengths,
                                          b0_gs_ns_ks_strides,
                                          b1_gs_os_ns_lengths,
                                          b1_gs_os_ns_strides,
                                          c_gs_ms_os_lengths,
                                          c_gs_ms_os_strides,
                                          {},             // acc0_biases_gs_ms_ns_lengths
                                          {},             // acc0_biases_gs_ms_ns_strides
                                          {},             // acc1_biases_gs_ms_os_lengths
                                          {},             // acc1_biases_gs_ms_os_strides
                                          PassThrough{},  // a_element_op
                                          PassThrough{},  // b0_element_op
                                          Scale{1.f},     // acc0_element_op
                                          PassThrough{},  // b1_element_op
                                          PassThrough{}); // c_element_op

        return gemm.IsSupportedArgument(argument);
    }
};
