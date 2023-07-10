// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/device_elementwise_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/device_gemm_mean_squaremean_instance.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType            = F16;
using BDataType            = F16;
using BiasDataType         = F32;
using CDataType            = F16;
using D0DataType           = F16;
using ReduceDataType       = F32;
using GammaDataType        = F16;
using BetaDataType         = F16;
using LayerNormOutDataType = F16;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

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

template <typename gemm_reduce_op_ptr>
bool RunDeviceGemmMeanSquareMean(gemm_reduce_op_ptr& p_op,
                                 const void* p_a,
                                 const void* p_b,
                                 const void* p_bias,
                                 const void* p_d0,
                                 void* p_c,
                                 void* p_mean,
                                 void* p_square_mean,
                                 int M,
                                 int N,
                                 int K,
                                 int StrideA,
                                 int StrideB,
                                 int StrideC,
                                 int StrideD0,
                                 bool time_kernel)
{
    using PassThrough          = ck::tensor_operation::element_wise::PassThrough;
    using UnaryDivElementOp    = ck::tensor_operation::element_wise::UnaryDivide;
    using UnarySquareElementOp = ck::tensor_operation::element_wise::UnarySquare;

    auto passOp   = PassThrough{};
    auto squareOp = UnarySquareElementOp{};
    auto divOp    = UnaryDivElementOp{N};

    auto argument_ptr =
        p_op->MakeArgumentPointer(p_a,
                                  p_b,
                                  p_bias,
                                  {p_d0},
                                  p_c,
                                  {p_mean, p_square_mean},
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC,
                                  {StrideD0},
                                  {&passOp, &passOp, &passOp}, // functor for a, b, c
                                  {&passOp},                   // functor for d0
                                  {&passOp, &squareOp},        // functor for inputs of reduction
                                  {&divOp, &divOp});           // functor for outputs of reduction

    if(p_op->IsSupportedArgument(argument_ptr.get()))
    {
        auto invoker_ptr = p_op->MakeInvokerPointer();

        // If we evaluate running time of gemm_reduce. The output may wrong.
        // Because we need to initialize the reduction tensor before runing the kernel.
        // However we run kernel many times for time_kernel = trie without reinitialize the out
        // of reduction tensor.
        float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            std::cout << "Gemm + reduce Perf: " << std::setw(10) << ave_time << " ms" << std::endl;

        return true;
    }

    return false;
}

template <typename normalize_op_ptr>
bool RunDeviceNormalize2D(normalize_op_ptr& p_op,
                          const void* p_x,
                          const void* p_mean,
                          const void* p_square_mean,
                          const void* p_gamma,
                          const void* p_beta,
                          void* p_y,
                          int M,
                          int N,
                          int StrideX,
                          bool time_kernel)
{
    std::array<const void*, 5> input = {p_x, p_mean, p_square_mean, p_gamma, p_beta};
    std::array<void*, 1> output      = {p_y};
    auto normalize_functor           = ck::tensor_operation::element_wise::Normalize{};

    std::array<ck::index_t, 2> xyLengths = {M, N};
    std::array<ck::index_t, 2> xyStrides = {StrideX, 1};

    auto argument_ptr = p_op->MakeArgumentPointer(xyLengths,
                                                  {xyStrides, {1, 0}, {1, 0}, {0, 1}, {0, 1}},
                                                  {xyStrides},
                                                  input,
                                                  output,
                                                  ck::tensor_operation::element_wise::Normalize{});

    if(p_op->IsSupportedArgument(argument_ptr.get()))
    {
        auto invoker_ptr = p_op->MakeInvokerPointer();
        float ave_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            std::cout << "Normalize Perf: " << std::setw(10) << ave_time << " ms" << std::endl;

        return true;
    }

    return false;
}

int main()
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideC  = 1024;
    ck::index_t StrideD0 = 1024;

    const auto gemm_reduce_ptrs =
        ck::tensor_operation::device::instance::get_device_gemm_add_add_mean_squaremean_instances<
            ADataType,
            BDataType,
            CDataType,
            ALayout,
            BLayout,
            CLayout>();

    const auto normalize_ptrs =
        ck::tensor_operation::device::instance::get_device_normalize_from_mean_meansquare_instances<
            CDataType,
            ReduceDataType,
            ReduceDataType,
            GammaDataType,
            BetaDataType,
            LayerNormOutDataType>();

    std::cout << "found " << gemm_reduce_ptrs.size()
              << " gemm_reduceMean_reduceSquareMean instances" << std::endl;

    std::cout << "found " << normalize_ptrs.size() << " normalize instances" << std::endl;

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    SimpleDeviceMem bias_device_buf(sizeof(BiasDataType) * N);
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(M, N, StrideC, CLayout{}));
    SimpleDeviceMem d0_device_buf(sizeof(D0DataType) *
                                  f_matrix_space_size(M, N, StrideD0, CLayout{}));
    SimpleDeviceMem reduceMean_device_buf(sizeof(ReduceDataType) * M);
    SimpleDeviceMem reduceMeanSquare_device_buf(sizeof(ReduceDataType) * M);
    SimpleDeviceMem gamma_device_buf(sizeof(GammaDataType) * N);
    SimpleDeviceMem beta_device_buf(sizeof(BetaDataType) * N);
    SimpleDeviceMem layerNorm_device_buf(sizeof(LayerNormOutDataType) * M * N);

    bool b_time_kernel           = true;
    bool b_only_run_first_kernel = true;

    // layernorm => (1) + (2)
    // (1). c = gemm(a, b), reduce_mean(c), reduce_square_mean(c)
    // (2). normalize(c, mean, square_mean, gamma, beta)
    for(auto& gemm_reduce_ptr : gemm_reduce_ptrs)
    {
        // run first available kernel
        if(RunDeviceGemmMeanSquareMean(gemm_reduce_ptr,
                                       a_device_buf.GetDeviceBuffer(),
                                       b_device_buf.GetDeviceBuffer(),
                                       bias_device_buf.GetDeviceBuffer(),
                                       d0_device_buf.GetDeviceBuffer(),
                                       c_device_buf.GetDeviceBuffer(),
                                       reduceMean_device_buf.GetDeviceBuffer(),
                                       reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                       M,
                                       N,
                                       K,
                                       StrideA,
                                       StrideB,
                                       StrideC,
                                       StrideD0,
                                       b_time_kernel))
        {
            if(b_only_run_first_kernel)
                break;
        }
        else
        {
            std::cout << gemm_reduce_ptr->GetTypeString() << " does not support this problem"
                      << std::endl;
        }
    }

    for(auto& normalize_ptr : normalize_ptrs)
    {
        if(RunDeviceNormalize2D(normalize_ptr,
                                c_device_buf.GetDeviceBuffer(),
                                reduceMean_device_buf.GetDeviceBuffer(),
                                reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                gamma_device_buf.GetDeviceBuffer(),
                                beta_device_buf.GetDeviceBuffer(),
                                layerNorm_device_buf.GetDeviceBuffer(),
                                M,
                                N,
                                StrideC,
                                b_time_kernel))
        {
            if(b_only_run_first_kernel)
                break;
        }
        else
        {
            std::cout << normalize_ptr->GetTypeString() << " does not support this problem"
                      << std::endl;
        }
    }
}
