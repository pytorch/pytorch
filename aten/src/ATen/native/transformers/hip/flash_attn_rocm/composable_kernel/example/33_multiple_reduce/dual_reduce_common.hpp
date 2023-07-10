// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>
#include <vector>
#include <array>
#include <algorithm>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths = {600, 28, 28, 256};
    size_t n, h, w, c;

    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = true;

    public:
    SimpleAppArgs()
    {
        n = inLengths[0];
        h = inLengths[1];
        w = inLengths[2];
        c = inLengths[3];
    };

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
                if(inLengths.size() != 4)
                    throw std::runtime_error(
                        "Invalid option format! The number of integers is incorrect!");

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

        n = inLengths[0];
        h = inLengths[1];
        w = inLengths[2];
        c = inLengths[3];

        return (0);
    };
};

template <typename InDataType, typename OutDataType1, typename OutDataType2, typename AccDataType>
static void mean_meansquare_host(const Tensor<InDataType>& in,
                                 Tensor<OutDataType1>& mean_ref,
                                 Tensor<OutDataType2>& meansquare_ref,
                                 size_t n,
                                 size_t h,
                                 size_t w,
                                 size_t c)

{
    auto thread_reduce_func = [&](auto iN) {
        AccDataType mean       = ck::type_convert<AccDataType>(0.0f);
        AccDataType meansquare = ck::type_convert<AccDataType>(0.0f);

        // compute mean, meanquare, variance, invVariance
        for(std::size_t iH = 0; iH < h; iH++)
        {
            for(std::size_t iW = 0; iW < w; iW++)
            {
                for(std::size_t iC = 0; iC < c; iC++)
                {
                    AccDataType curr_value = ck::type_convert<AccDataType>(in(iN, iH, iW, iC));

                    mean += curr_value;
                    meansquare += curr_value * curr_value;
                };
            }
        };

        mean       = mean / (h * w * c);
        meansquare = meansquare / (h * w * c);

        mean_ref(iN)       = ck::type_convert<OutDataType1>(mean);
        meansquare_ref(iN) = ck::type_convert<OutDataType2>(meansquare);
    };

    std::size_t num_thread      = std::thread::hardware_concurrency();
    std::size_t work_per_thread = (n + num_thread - 1) / num_thread;

    std::vector<joinable_thread> threads(num_thread);

    for(std::size_t it = 0; it < num_thread; it++)
    {
        std::size_t iN_begin = it * work_per_thread;
        std::size_t iN_end   = std::min(static_cast<size_t>((it + 1) * work_per_thread), n);

        auto f = [=] {
            for(std::size_t iN = iN_begin; iN < iN_end; iN++)
            {
                thread_reduce_func(iN);
            }
        };

        threads[it] = joinable_thread(f);
    }
};

using ReduceOperation = ck::reduce::Add;

using InElementwiseOperation_Mean  = ck::tensor_operation::element_wise::PassThrough;
using AccElementwiseOperation_Mean = ck::tensor_operation::element_wise::UnaryDivide;

using InElementwiseOperation_Meansquare  = ck::tensor_operation::element_wise::UnarySquare;
using AccElementwiseOperation_Meansquare = ck::tensor_operation::element_wise::UnaryDivide;

using InElementwiseOperationTuple =
    ck::Tuple<InElementwiseOperation_Mean, InElementwiseOperation_Meansquare>;
using AccElementwiseOperationTuple =
    ck::Tuple<AccElementwiseOperation_Mean, AccElementwiseOperation_Meansquare>;

template <typename DeviceDualReduce,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          int Rank,
          int NumReduceDim>
int mean_meansquare_dual_reduce_test(size_t n,
                                     size_t h,
                                     size_t w,
                                     size_t c,
                                     bool do_verification,
                                     int init_method,
                                     bool time_kernel,
                                     const std::array<int, NumReduceDim> reduceDims)
{
    const std::vector<size_t> inLengths = {n, h, w, c};

    Tensor<InDataType> in(inLengths);

    std::vector<size_t> outLengths{n};

    Tensor<OutDataType> mean_ref(outLengths);
    Tensor<OutDataType> mean(outLengths);
    Tensor<OutDataType> meansquare_ref(outLengths);
    Tensor<OutDataType> meansquare(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = mean.mDesc.GetStrides();

    size_t invariant_total_length = n;
    size_t reduce_total_length    = h * w * c;

    const AccDataType alpha = ck::type_convert<AccDataType>(1.0f);
    const AccDataType beta  = ck::type_convert<AccDataType>(0.0f);

    std::size_t num_thread = 1;

    if(do_verification)
    {
        switch(init_method)
        {
        case 0: break;
        case 1: in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread); break;
        case 2: in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread); break;
        default: in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem mean_dev(sizeof(OutDataType) * mean.mDesc.GetElementSpaceSize());
    DeviceMem meansquare_dev(sizeof(OutDataType) * meansquare.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());

    if(do_verification)
    {
        mean_meansquare_host<InDataType, OutDataType, OutDataType, AccDataType>(
            in, mean_ref, meansquare_ref, n, h, w, c);
    };

    constexpr ck::index_t NumInputDim  = Rank;
    constexpr ck::index_t NumOutputDim = (Rank - NumReduceDim > 1) ? Rank - NumReduceDim : 1;

    std::array<ck::index_t, NumInputDim> i_inLengths;
    std::array<ck::index_t, NumInputDim> i_inStrides;
    std::array<ck::index_t, NumOutputDim> i_outLengths;
    std::array<ck::index_t, NumOutputDim> i_outStrides;

    ck::ranges::copy(inLengths, i_inLengths.begin());
    ck::ranges::copy(inStrides, i_inStrides.begin());
    ck::ranges::copy(outLengths, i_outLengths.begin());
    ck::ranges::copy(outStrides, i_outStrides.begin());

    auto dual_reduce_op = DeviceDualReduce{};

    auto argument_ptr = dual_reduce_op.MakeArgumentPointer(
        i_inLengths,
        i_inStrides,
        i_outLengths,
        {i_outStrides, i_outStrides},
        reduceDims,
        {&alpha, &alpha},
        {&beta, &beta},
        in_dev.GetDeviceBuffer(),
        {mean_dev.GetDeviceBuffer(), meansquare_dev.GetDeviceBuffer()},
        ck::make_tuple(InElementwiseOperation_Mean{}, InElementwiseOperation_Meansquare{}),
        ck::make_tuple(
            AccElementwiseOperation_Mean{static_cast<int32_t>(reduce_total_length)},
            AccElementwiseOperation_Meansquare{static_cast<int32_t>(reduce_total_length)}));

    if(!dual_reduce_op.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
        return (-1);
    };

    std::string reduce_name = dual_reduce_op.GetTypeString();

    auto invoker_ptr = dual_reduce_op.MakeInvokerPointer();

    float avg_time = 0.0f;

    avg_time += invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InDataType) +
                            2 * invariant_total_length * sizeof(OutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        mean_dev.FromDevice(mean.mData.data());
        meansquare_dev.FromDevice(meansquare.mData.data());
        pass = pass && ck::utils::check_err(mean, mean_ref);
        pass = pass && ck::utils::check_err(meansquare, meansquare_ref);
    };

    return (pass ? 0 : 1);
}
