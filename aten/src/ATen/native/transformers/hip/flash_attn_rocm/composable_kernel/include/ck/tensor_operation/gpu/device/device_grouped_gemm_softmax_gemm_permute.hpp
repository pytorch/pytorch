// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

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
struct DeviceGroupedGemmSoftmaxGemmPermute : public BaseOperator
{
    struct ProblemDesc
    {
        std::vector<index_t> a_gs_ms_ks_lengths;
        std::vector<index_t> a_gs_ms_ks_strides;

        std::vector<index_t> b0_gs_ns_ks_lengths;
        std::vector<index_t> b0_gs_ns_ks_strides;

        std::vector<index_t> b1_gs_os_ns_lengths;
        std::vector<index_t> b1_gs_os_ns_strides;

        std::vector<index_t> c_gs_ms_os_lengths;
        std::vector<index_t> c_gs_ms_os_strides;

        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_lengths;
        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_strides;

        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_lengths;
        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_strides;
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b0_vec,
                        std::vector<const void*> p_b1_vec,
                        std::vector<void*> p_c_vec,
                        std::vector<std::vector<const void*>> p_acc0_biases_vec,
                        std::vector<std::vector<const void*>> p_acc1_biases_vec,
                        std::vector<ProblemDesc> problem_desc_vec,
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
struct DeviceGroupedMultiheadAttentionForward : public BaseOperator
{
    struct ProblemDesc
    {
        std::vector<index_t> a_gs_ms_ks_lengths;
        std::vector<index_t> a_gs_ms_ks_strides;

        std::vector<index_t> b0_gs_ns_ks_lengths;
        std::vector<index_t> b0_gs_ns_ks_strides;

        std::vector<index_t> b1_gs_os_ns_lengths;
        std::vector<index_t> b1_gs_os_ns_strides;

        std::vector<index_t> c_gs_ms_os_lengths;
        std::vector<index_t> c_gs_ms_os_strides;

        std::vector<index_t> z_gs_ms_ns_lengths;
        std::vector<index_t> z_gs_ms_ns_strides;

        std::vector<index_t> lse_gs_ms_lengths;
        std::vector<index_t> lse_gs_ms_strides;

        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_lengths;
        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_strides;

        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_lengths;
        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_strides;
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b0_vec,
                        std::vector<const void*> p_b1_vec,
                        std::vector<void*> p_c_vec,
                        std::vector<void*> p_z_vec,
                        std::vector<void*> p_lse_vec,
                        std::vector<std::vector<const void*>> p_acc0_biases_vec,
                        std::vector<std::vector<const void*>> p_acc1_biases_vec,
                        std::vector<ProblemDesc> problem_desc_vec,
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
