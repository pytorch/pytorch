// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <tuple>

#include "device_base.hpp"
#include "ck/tensor_operation/gpu/device/masking_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename Acc0BiasDataType,
          typename Acc1BiasDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          MaskingSpecialization MaskingSpec>
struct DeviceBatchedGemmSoftmaxGemmPermute : public BaseOperator
{
    static constexpr index_t NumAcc0Bias = Acc0BiasDataType::Size();
    static constexpr index_t NumAcc1Bias = Acc1BiasDataType::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b0,
        const void* p_b1,
        void* p_c,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b_gs_ns_ks_lengths,
        const std::vector<index_t>& b_gs_ns_ks_strides,
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
        const std::vector<index_t>& c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
        const std::vector<index_t>& c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
        const std::array<std::vector<index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_lengths,
        const std::array<std::vector<index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_strides,
        const std::array<std::vector<index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
        const std::array<std::vector<index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
        AElementwiseOperation a_element_op,
        B0ElementwiseOperation b0_element_op,
        Acc0ElementwiseOperation acc0_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ZDataType,
          typename LSEDataType,
          typename Acc0BiasDataType,
          typename Acc1BiasDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          MaskingSpecialization MaskingSpec>
struct DeviceBatchedMultiheadAttentionForward : public BaseOperator
{
    static constexpr index_t NumAcc0Bias = Acc0BiasDataType::Size();
    static constexpr index_t NumAcc1Bias = Acc1BiasDataType::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b0,
        const void* p_b1,
        void* p_c,
        void* p_z,
        void* p_lse,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b_gs_ns_ks_lengths,
        const std::vector<index_t>& b_gs_ns_ks_strides,
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
        const std::vector<index_t>& c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
        const std::vector<index_t>& c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
        const std::vector<index_t>& z_gs_ms_ns_lengths,            // z_gs_ms_os_lengths
        const std::vector<index_t>& z_gs_ms_ns_strides,            // z_gs_ms_os_strides
        const std::vector<index_t>& lse_gs_ms_lengths,             // lse_gs_ms_lengths
        const std::array<std::vector<index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_lengths,
        const std::array<std::vector<index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_strides,
        const std::array<std::vector<index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
        const std::array<std::vector<index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
        AElementwiseOperation a_element_op,
        B0ElementwiseOperation b0_element_op,
        Acc0ElementwiseOperation acc0_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op,
        float p_dropout,
        std::tuple<unsigned long long, unsigned long long> seeds) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
