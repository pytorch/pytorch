// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "host_tensor.hpp"

template <typename AType,
          typename BType,
          typename CType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void host_gemm_mk_kn_mn(const Tensor<AType>& a_m_k,
                        const Tensor<BType>& b_k_n,
                        Tensor<CType>& c_m_n,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CElementwiseOperation& c_element_op)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        float v_acc = 0;

        for(int k = 0; k < K; ++k)
        {
            float v_a;
            float v_b;

            a_element_op(v_a, static_cast<const float>(a_m_k(m, k)));
            b_element_op(v_b, static_cast<const float>(b_k_n(k, n)));

            v_acc += v_a * v_b;
        }

        float v_c;

        c_element_op(v_c, v_acc);

        c_m_n(m, n) = v_c;
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}
