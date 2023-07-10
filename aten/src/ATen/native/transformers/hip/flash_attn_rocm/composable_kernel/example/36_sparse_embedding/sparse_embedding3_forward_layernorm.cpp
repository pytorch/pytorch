// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>
#include <ctime>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_sparse_embedding3_forward_layernorm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_sparse_embedding3_forward_layernorm.hpp"

// using EmbType       = float;
// using IndexType     = int64_t;
// using GammaDataType = float;
// using BetaDataType  = float;
// using AccDataType   = float;
// using OutType       = float;

using EmbType       = ck::half_t;
using IndexType     = int64_t;
using GammaDataType = ck::half_t;
using BetaDataType  = ck::half_t;
using AccDataType   = float;
using OutType       = ck::half_t;

// clang-format off
//                                                                                                         BlockSize, DimClusterSize, RowClusterSize, DimPerBlock, RowPerBlock, DimThreadSize, RowVectorSize
using DeviceInstance_fp32_e256   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  256,   1, 1>;
using DeviceInstance_fp32_e512   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  512,   1, 1>;
using DeviceInstance_fp32_e768   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  768,   1, 1>;
using DeviceInstance_fp32_e1024  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  1024,  1, 1>;
using DeviceInstance_fp32_e1536  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  1536,  1, 1>;
using DeviceInstance_fp32_e2048  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  2048,  1, 4>;
using DeviceInstance_fp32_e4096  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  4096,  1, 4>;
using DeviceInstance_fp32_e8192  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  8192,  1, 4>;
using DeviceInstance_fp32_e16384 = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  16384, 1, 4>;

using DeviceInstance_fp16_e256   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  256,   1, 1>;
using DeviceInstance_fp16_e512   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  512,   1, 2>;
using DeviceInstance_fp16_e768   = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  768,   1, 1>;
using DeviceInstance_fp16_e1024  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  1024,  1, 2>;
using DeviceInstance_fp16_e1536  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  1536,  1, 2>;
using DeviceInstance_fp16_e2048  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  2048,  1, 2>;
using DeviceInstance_fp16_e4096  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  4096,  1, 8>;
using DeviceInstance_fp16_e8192  = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<EmbType, IndexType, GammaDataType, BetaDataType, AccDataType, OutType, 256,  1,  256, 1,  8192,  1, 8>;

template<typename emb_type, ck::index_t dim> struct emb_kernel{};

template<> struct emb_kernel<float, 256>  { using kernel_type = DeviceInstance_fp32_e256; };
template<> struct emb_kernel<float, 512>  { using kernel_type = DeviceInstance_fp32_e512; };
template<> struct emb_kernel<float, 768>  { using kernel_type = DeviceInstance_fp32_e768; };
template<> struct emb_kernel<float, 1024> { using kernel_type = DeviceInstance_fp32_e1024;};
template<> struct emb_kernel<float, 1536> { using kernel_type = DeviceInstance_fp32_e1536;};
template<> struct emb_kernel<float, 2048> { using kernel_type = DeviceInstance_fp32_e2048;};
template<> struct emb_kernel<float, 4096> { using kernel_type = DeviceInstance_fp32_e4096;};
template<> struct emb_kernel<float, 8192> { using kernel_type = DeviceInstance_fp32_e8192;};
template<> struct emb_kernel<float, 16384>{ using kernel_type = DeviceInstance_fp32_e16384;};

template<> struct emb_kernel<ck::half_t, 256>  { using kernel_type = DeviceInstance_fp16_e256; };
template<> struct emb_kernel<ck::half_t, 512>  { using kernel_type = DeviceInstance_fp16_e512; };
template<> struct emb_kernel<ck::half_t, 768>  { using kernel_type = DeviceInstance_fp16_e768; };
template<> struct emb_kernel<ck::half_t, 1024> { using kernel_type = DeviceInstance_fp16_e1024; };
template<> struct emb_kernel<ck::half_t, 1536> { using kernel_type = DeviceInstance_fp16_e1536; };
template<> struct emb_kernel<ck::half_t, 2048> { using kernel_type = DeviceInstance_fp16_e2048; };
template<> struct emb_kernel<ck::half_t, 4096> { using kernel_type = DeviceInstance_fp16_e4096; };
template<> struct emb_kernel<ck::half_t, 8192> { using kernel_type = DeviceInstance_fp16_e8192; };

// clang-format on

int main()
{
    bool time_kernel = true;

    constexpr auto num_rows = 65536;
    constexpr auto dims     = ck::Sequence<256, 512, 768, 1024, 1536, 2048, 4096, 8192>{};
    // constexpr auto dims = ck::Sequence<256, 512>{};
    constexpr auto index_length   = 2048;
    constexpr AccDataType epsilon = 1e-4;

    auto f_host_tensor_desc_1d = [](std::size_t len_) { return HostTensorDescriptor({len_}); };

    auto f_host_tensor_desc_2d = [](std::size_t rows_, std::size_t cols_) {
        return HostTensorDescriptor({rows_, cols_});
    };

    using ReferenceInstance =
        ck::tensor_operation::host::ReferenceSparseEmbedding3ForwardLayernorm<EmbType,
                                                                              IndexType,
                                                                              GammaDataType,
                                                                              BetaDataType,
                                                                              AccDataType,
                                                                              OutType>;

    ck::static_for<0, dims.Size(), 1>{}([&](auto I) {
        std::srand(std::time(nullptr));
        constexpr auto current_dim = dims.At(I);
        Tensor<EmbType> emb_a(f_host_tensor_desc_2d(num_rows, current_dim));
        Tensor<EmbType> emb_b(f_host_tensor_desc_2d(num_rows, current_dim));
        Tensor<EmbType> emb_c(f_host_tensor_desc_2d(num_rows, current_dim));

        Tensor<IndexType> index_a(f_host_tensor_desc_1d(index_length));
        Tensor<IndexType> index_b(f_host_tensor_desc_1d(index_length));
        Tensor<IndexType> index_c(f_host_tensor_desc_1d(index_length));

        Tensor<GammaDataType> gamma(f_host_tensor_desc_1d(current_dim));
        Tensor<BetaDataType> beta(f_host_tensor_desc_1d(current_dim));

        Tensor<OutType> out(f_host_tensor_desc_2d(index_length, current_dim));

        emb_a.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
        emb_b.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
        emb_c.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});

        index_a.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});
        index_b.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});
        index_c.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});

        gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
        beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{0.0, 1.0});

        DeviceMem emb_a_dev(sizeof(EmbType) * emb_a.mDesc.GetElementSpaceSize());
        DeviceMem emb_b_dev(sizeof(EmbType) * emb_b.mDesc.GetElementSpaceSize());
        DeviceMem emb_c_dev(sizeof(EmbType) * emb_c.mDesc.GetElementSpaceSize());

        DeviceMem index_a_dev(sizeof(IndexType) * index_a.mDesc.GetElementSpaceSize());
        DeviceMem index_b_dev(sizeof(IndexType) * index_b.mDesc.GetElementSpaceSize());
        DeviceMem index_c_dev(sizeof(IndexType) * index_c.mDesc.GetElementSpaceSize());

        DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
        DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());

        DeviceMem out_dev(sizeof(OutType) * out.mDesc.GetElementSpaceSize());

        emb_a_dev.ToDevice(emb_a.mData.data());
        emb_b_dev.ToDevice(emb_b.mData.data());
        emb_c_dev.ToDevice(emb_c.mData.data());

        index_a_dev.ToDevice(index_a.mData.data());
        index_b_dev.ToDevice(index_b.mData.data());
        index_c_dev.ToDevice(index_c.mData.data());

        gamma_dev.ToDevice(gamma.mData.data());
        beta_dev.ToDevice(beta.mData.data());

        auto device_instance = typename emb_kernel<EmbType, current_dim>::kernel_type{};
        auto argument_ptr    = device_instance.MakeArgumentPointer(out_dev.GetDeviceBuffer(),
                                                                emb_a_dev.GetDeviceBuffer(),
                                                                emb_b_dev.GetDeviceBuffer(),
                                                                emb_c_dev.GetDeviceBuffer(),
                                                                index_a_dev.GetDeviceBuffer(),
                                                                index_b_dev.GetDeviceBuffer(),
                                                                index_c_dev.GetDeviceBuffer(),
                                                                gamma_dev.GetDeviceBuffer(),
                                                                beta_dev.GetDeviceBuffer(),
                                                                num_rows,
                                                                current_dim,
                                                                index_length,
                                                                epsilon);
        std::cout << "Dim:" << current_dim << ", kernel:" << device_instance.GetTypeString()
                  << std::endl
                  << std::flush;

        bool is_supported = device_instance.IsSupportedArgument(argument_ptr.get());

        if(!is_supported)
        {
            std::cout << "Runtime parameters are not supported" << std::endl;
            return;
        }

        auto invoker_ptr = device_instance.MakeInvokerPointer();
        float time_ms    = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        bool pass = true;
        {
            Tensor<OutType> out_from_dev(f_host_tensor_desc_2d(index_length, current_dim));
            ReferenceInstance ref;
            auto ref_argument = ref.MakeArgument(out,
                                                 emb_a,
                                                 emb_b,
                                                 emb_c,
                                                 index_a,
                                                 index_b,
                                                 index_c,
                                                 gamma,
                                                 beta,
                                                 num_rows,
                                                 current_dim,
                                                 index_length,
                                                 epsilon);
            auto ref_invoker  = ref.MakeInvoker();
            ref_invoker.Run(ref_argument);

            out_dev.FromDevice(out_from_dev.mData.data());
            pass &= ck::utils::check_err(out_from_dev, out, "Error: Incorrect results", 1e-3, 1e-3);
        }

        double total_read = current_dim * index_length * 3 * sizeof(EmbType) +
                            current_dim * sizeof(GammaDataType) +
                            current_dim * sizeof(BetaDataType);
        double total_write = current_dim * index_length * sizeof(OutType);
        double gbps        = (total_read + total_write) / time_ms / 1e6;

        std::cout << ", total bytes:" << (total_read + total_write) << ", time:" << time_ms
                  << ", gbps:" << gbps << ", valid:" << (pass ? "y" : "n") << std::endl
                  << std::flush;
    });

    return 0;
}
