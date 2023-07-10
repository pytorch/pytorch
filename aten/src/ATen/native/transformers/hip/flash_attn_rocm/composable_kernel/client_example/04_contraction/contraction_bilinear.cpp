// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <numeric>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/contraction_bilinear.hpp"
#include "ck/library/utility/numeric.hpp"

using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Bilinear;

using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F32;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F32;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

int main(int argc, char* argv[])
{
    // A[M0, M1, K0, K1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // B[N0, N1, K0, K1]
    std::vector<ck::index_t> b_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ns_ks_strides{524288, 4096, 128, 1};
    // D[M0, M1, N0, N1]
    std::vector<ck::index_t> d_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> d_ms_ns_strides{524288, 4096, 128, 1};
    // E[M0, M1, N0, N1]
    std::vector<ck::index_t> e_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> e_ms_ns_strides{524288, 4096, 128, 1};

    float alpha = 1.f;
    float beta  = 1.f;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 25)
    {
        const ck::index_t M0 = std::stoi(argv[1]);
        const ck::index_t M1 = std::stoi(argv[2]);

        const ck::index_t N0 = std::stoi(argv[3]);
        const ck::index_t N1 = std::stoi(argv[4]);

        const ck::index_t K0 = std::stoi(argv[5]);
        const ck::index_t K1 = std::stoi(argv[6]);

        a_ms_ks_lengths = {M0, M1, K0, K1};
        a_ms_ks_strides = {
            std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[9]), std::stoi(argv[10])};

        b_ns_ks_lengths = {N0, N1, K0, K1};
        b_ns_ks_strides = {
            std::stoi(argv[11]), std::stoi(argv[12]), std::stoi(argv[13]), std::stoi(argv[14])};

        d_ms_ns_lengths = {M0, M1, N0, N1};
        d_ms_ns_strides = {
            std::stoi(argv[15]), std::stoi(argv[16]), std::stoi(argv[17]), std::stoi(argv[18])};

        e_ms_ns_lengths = {M0, M1, N0, N1};
        e_ms_ns_strides = {
            std::stoi(argv[19]), std::stoi(argv[20]), std::stoi(argv[21]), std::stoi(argv[22])};

        alpha = std::stof(argv[23]);
        beta  = std::stof(argv[24]);
    }
    else
    {
        printf("arg1 to 6: M0, M1, N0, N1, K0, K1\n");
        printf("arg7 to 10: Stride_A_M0, Stride_A_M1, Stride_A_K0, Stride_A_K1\n");
        printf("arg11 to 14: Stride_B_N0, Stride_B_N1, Stride_B_K0, Stride_B_K1\n");
        printf("arg15 to 18: Stride_D_M0, Stride_D_M1, Stride_D_N0, Stride_D_N1\n");
        printf("arg19 to 22: Stride_E_M0, Stride_E_M1, Stride_E_N0, Stride_E_N1\n");
        printf("arg23 to 24: alpha, beta\n");
        exit(0);
    }

    auto f_tensor_space_size = [](auto lengths, auto strides) {
        std::size_t space_size = 1;
        for(std::size_t i = 0; i < lengths.size(); ++i)
        {
            space_size += (lengths[i] - 1) * strides[i];
        }
        return space_size;
    };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) *
                                 f_tensor_space_size(a_ms_ks_lengths, a_ms_ks_strides));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) *
                                 f_tensor_space_size(b_ns_ks_lengths, b_ns_ks_strides));
    SimpleDeviceMem d_device_buf(sizeof(DDataType) *
                                 f_tensor_space_size(d_ms_ns_lengths, d_ms_ns_strides));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) *
                                 f_tensor_space_size(e_ms_ns_lengths, e_ms_ns_strides));

    using DeviceOp = ck::tensor_operation::device::DeviceContractionMultipleD<
        NumDimM,
        NumDimN,
        NumDimK,
        ADataType,
        BDataType,
        ck::Tuple<DDataType>,
        EDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Bilinear>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{alpha, beta};

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr =
            op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                        b_device_buf.GetDeviceBuffer(),
                                        std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                                        e_device_buf.GetDeviceBuffer(),
                                        a_ms_ks_lengths,
                                        a_ms_ks_strides,
                                        b_ns_ks_lengths,
                                        b_ns_ks_strides,
                                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                                        e_ms_ns_lengths,
                                        e_ms_ns_strides,
                                        a_element_op,
                                        b_element_op,
                                        cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            ck::index_t M = ck::accumulate_n<ck::index_t>(
                e_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

            ck::index_t N = ck::accumulate_n<ck::index_t>(
                e_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

            ck::index_t K = ck::accumulate_n<ck::index_t>(
                a_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

            std::size_t flop      = std::size_t(2) * M * N * K;
            std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                    sizeof(DDataType) * M * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return 0;
}
