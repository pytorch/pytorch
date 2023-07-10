// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace ck {
namespace wmma_op_util {

template <typename src_vec, typename acc_vec>
__device__ void builtin_wmma_naive_selector(const src_vec&, const src_vec&, acc_vec&)
{
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_bf16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>& reg_c)
{
    intrin_wmma_f16_16x16x16_f16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_wmma_naive_selector<
    bhalf16_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>& reg_c)
{
    intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<int8x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int8x16_t& reg_a,
    const int8x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu8_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
__device__ void
builtin_wmma_naive_selector<int4x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int4x16_t& reg_a,
    const int4x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu4_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}
#endif

template <typename src_t, typename dst_t, typename acc_t, index_t acc_num>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
    const int lIdx = threadIdx.x;
    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and
    // b a_frag will store one column of the 16x16 matrix tile b_frag will store one row of the
    // 16x16 matrix tile
    using src_vec  = typename vector_type<src_t, 16>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in gfx11
    // see https://atlvsp3.amd.com/sp3_gfx11_5_instructions.pdf page 482
    // TODO: remove this dependency in gfx12 https://ontrack-internal.amd.com/browse/DEGFXSP3-101
    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }
    // follow origin design
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // sync threads, similar to mma_sync
    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();
    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

template <typename src_t, typename dst_t, typename acc_t, index_t acc_num>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
    const int lIdx = threadIdx.x;

    using src_vec  = typename vector_type<src_t, 16>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};
    using acc_vec  = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }

    const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * offset_m + ele];
    }

    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();

    static_for<0, 8, 1>{}([&](auto ele) {
        const int blk = lIdx / 16;
        const int r   = ele;
        c[16 * 8 * blk + 16 * r + lane] =
            ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

struct GemmParams
{
    GemmParams() : M(16), N(16), K(16), StrideA(16), StrideB(16), StrideC(16), alpha(1), beta(0) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceWmma,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t CAccNum>
struct TestWmma
{
    auto PrepareGemmTensor(const ck::wmma_op_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_n_k, BDataType{});

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceWmma& wmma_kernel)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;

        // Arrange
        ck::wmma_op_util::GemmParams params;
        params.M       = 16;
        params.N       = 16;
        params.K       = 16;
        params.StrideA = 16;
        params.StrideB = 16;
        params.StrideC = 16;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CPUAccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::wmma_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = ck::wmma_op_util::RunDeviceGEMM(wmma_kernel, a, b, c_device);

        if(is_supported)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::half_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::bhalf_t>::value)
            {
                // 0.5 Pixel Error Tolerance is introduced by Accumulator difference.
                // BF16 WMMA Accumulator is in BF16 Type while On Host-side Accumulator is Float.
                res = ck::utils::check_err(
                    c_device.mData, c_host.mData, "Error: Incorrect results!", 0, 1.0);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else
            {
                std::cout << "UNSUPPORTED CDataType" << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace wmma_op_util
} // namespace ck
