// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void add_gemm_f16_tt_256x256(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_f16_tt_256x128(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_f16_tt_128x128(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_f16_tt_128x64(std::vector<std::unique_ptr<BaseOperator>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
