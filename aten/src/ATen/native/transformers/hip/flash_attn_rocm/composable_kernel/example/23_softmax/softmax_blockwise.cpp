// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

using namespace ck::tensor_operation::device;

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 3;
constexpr int NumReduceDim = 1;

using DeviceInstance = DeviceSoftmaxImpl<InDataType,
                                         AccDataType,
                                         OutDataType,
                                         PassThrough, // InElementwiseOperation
                                         PassThrough, // AccElementwiseOperation
                                         Rank,
                                         NumReduceDim,
                                         256, // BlockSize
                                         8,   // ClusterM
                                         32,  // ClusterK
                                         1,   // SliceM
                                         8,   // SliceK
                                         1,   // SrcVecDim (0=M, 1=K)
                                         8,   // SrcScalarPerVector
                                         8>;  // OutScalarPerVector

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths   = {8, 128, 2048};
    std::vector<AccDataType> scales = {2.0f, 2.0f};

    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = true;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "Arg1 -- init method (0=no init, 1=single integer value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg2 -- time kernel (0=no, 1=yes)" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:l:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inLengths = getTypeValuesFromString<size_t>(optarg);
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;
            default: show_usage(argv[0]); return (-1);
            };
        };

        if(optind + 2 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        init_method = std::atoi(argv[optind++]);
        time_kernel = static_cast<bool>(std::atoi(argv[optind]));

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        return (0);
    };
};

int main(int argc, char* argv[])
{
    // Example: batched gemm C[G, M, N] applies max/sum reduction along N internally
    const std::vector<int> invariantDims{0, 1};
    const std::vector<int> reduceDims{2};

    SimpleAppArgs args;

    if(argc > 1)
    {
        if(args.processArgs(argc, argv) < 0)
            return (-1);
    };

    Tensor<InDataType> in(args.inLengths);
    Tensor<OutDataType> out_ref(args.inLengths);
    Tensor<OutDataType> out(args.inLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    AccDataType alpha = args.scales[0];
    AccDataType beta  = args.scales[1];

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    std::size_t num_thread = 1;

    if(args.do_verification)
    {
        switch(args.init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-5.0, 5.0}, num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                out.mData[i] = out_ref.mData[i];
    };
    // std::cout << "beta = " << beta << std::endl;
    // LogRangeAsType<float>(std::cout << "tensor in: " , in.mData, ",") << std::endl;
    // LogRangeAsType<float>(std::cout << "tensor prior out: " , out.mData, ",") << std::endl;

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    if(args.do_verification)
    {
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceSoftmax<InDataType, OutDataType, AccDataType>;
        ReferenceInstance ref;
        auto ref_arg = ref.MakeArgument(in, out_ref, alpha, beta, reduceDims);
        auto invoker = ref.MakeInvoker();
        invoker.Run(ref_arg);
        // LogRangeAsType<float>(std::cout << "tensor out_ref: ", out_ref.mData, ",") << std::endl;
    };

    std::vector<ck::index_t> i_inLengths;
    std::vector<ck::index_t> i_inStrides;

    i_inLengths.assign(args.inLengths.begin(), args.inLengths.end());
    i_inStrides.assign(inStrides.begin(), inStrides.end());

    auto device_instance = DeviceInstance{};

    std::cout << i_inLengths.size() << ", " << i_inStrides.size() << std::endl;

    auto argument_ptr = device_instance.MakeArgumentPointer(i_inLengths,
                                                            i_inStrides,
                                                            reduceDims,
                                                            &alpha,
                                                            &beta,
                                                            in_dev.GetDeviceBuffer(),
                                                            out_dev.GetDeviceBuffer(),
                                                            PassThrough{},
                                                            PassThrough{});

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
        return 1;
    };

    std::string instance_name = device_instance.GetTypeString();

    auto invoker_ptr = device_instance.MakeInvokerPointer();

    bool pass = true;
    if(args.do_verification)
    {
        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        out_dev.FromDevice(out.mData.data());
        // LogRangeAsType<float>(std::cout << "tensor out: " , out.mData, ",") << std::endl;
        pass = pass && ck::utils::check_err(out, out_ref);
    };

    float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, args.time_kernel});

    std::size_t num_bytes =
        in.mDesc.GetElementSize() * sizeof(InDataType) +
        (beta == 0.0f ? 1 : 2) * out.mDesc.GetElementSize() * sizeof(OutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << instance_name
              << std::endl;

    return (pass ? 0 : 1);
}
