// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <getopt.h>

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_batchnorm_forward_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

using namespace std;

static const struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                             {"reduceDims", required_argument, nullptr, 'R'},
                                             {"dumpout", required_argument, nullptr, 'o'},
                                             {"verify", required_argument, nullptr, 'v'},
                                             {"help", no_argument, nullptr, '?'},
                                             {nullptr, 0, nullptr, 0}};

class BatchnormFwdArgParser
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths;
    std::vector<int> reduceDims;

    bool do_verification = false;
    bool do_dumpout      = false;

    bool updateMovingAverage;
    bool saveMeanAndInvVariance;

    int data_type    = 0;
    int init_method  = 2;
    bool time_kernel = false;

    BatchnormFwdArgParser()  = default;
    ~BatchnormFwdArgParser() = default;

    void show_usage(const char* cmd)
    {
        // clang-format off
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inOutLengths or -D, comma separated list of input tensor dimension lengths, must have 4 integers for nhwc" << std::endl;
        std::cout << "--reduceDims or -R, comma separated list of dimensions to reduce on" << std::endl;  
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the result by comparing with the host-based batch-normalization" << std::endl;
        std::cout << "Arg1: data type (0: fp16, 1: fp32, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2: 1/0 to indicate whether to update the moving average and variance (0=no, 1=yes)" << std::endl;
        std::cout << "Arg3: 1/0 to indicate whether to save the calculated mean and invVariance (0=no, 1=yes)" << std::endl;
        std::cout << "Arg4: init method used for bnScale and bnBias (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)" << std::endl;
        std::cout << "Arg5: time kernel (0=no, 1=yes)" << std::endl;
        // clang-format on
    };

    int operator()(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        optind++; // to skip the module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:v:o:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inLengths = getTypeValuesFromString<size_t>(optarg);
                break;
            case 'R':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                reduceDims = getTypeValuesFromString<int>(optarg);
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case 'o':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_dumpout = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return -1;
                };
                break;

            default:
                show_usage(argv[0]);
                std::cerr << "Invalid cmd-line options!" << std::endl;
                return -1;
            };
        };

        if(optind + 5 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        data_type              = std::atoi(argv[optind++]);
        updateMovingAverage    = std::atoi(argv[optind++]);
        saveMeanAndInvVariance = std::atoi(argv[optind++]);
        init_method            = std::atoi(argv[optind++]);
        time_kernel            = static_cast<bool>(std::atoi(argv[optind++]));

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5 && data_type != 6)
            return -1;

        return 0;
    };
}; // end of class AppArgs

static const double epsilon       = std::numeric_limits<float>::epsilon();
static const double averageFactor = 0.1;

int profile_batchnorm_forward(int argc, char* argv[])
{
    using ck::profiler::profile_batchnorm_forward_impl;

    BatchnormFwdArgParser arg_parser;

    if(arg_parser(argc, argv) != 0)
        return -1;

    using F16  = ck::half_t;
    using F32  = float;
    using BF16 = ck::bhalf_t;
    using F64  = double;

    if(arg_parser.data_type == 0)
    {
        if(arg_parser.inLengths.size() == 4 && arg_parser.reduceDims.size() == 3)
        {
            profile_batchnorm_forward_impl<F16, F16, F32, F16, F16, F16, 4, 3>(
                arg_parser.do_verification,
                arg_parser.init_method,
                arg_parser.do_dumpout,
                arg_parser.time_kernel,
                arg_parser.inLengths,
                arg_parser.reduceDims,
                arg_parser.updateMovingAverage,
                arg_parser.saveMeanAndInvVariance,
                epsilon,
                averageFactor);
        };
    }
    else if(arg_parser.data_type == 1)
    {
        if(arg_parser.inLengths.size() == 4 && arg_parser.reduceDims.size() == 3)
        {
            profile_batchnorm_forward_impl<F32, F32, F32, F32, F32, F32, 4, 3>(
                arg_parser.do_verification,
                arg_parser.init_method,
                arg_parser.do_dumpout,
                arg_parser.time_kernel,
                arg_parser.inLengths,
                arg_parser.reduceDims,
                arg_parser.updateMovingAverage,
                arg_parser.saveMeanAndInvVariance,
                epsilon,
                averageFactor);
        };
    }
    else if(arg_parser.data_type == 5)
    {
        if(arg_parser.inLengths.size() == 4 && arg_parser.reduceDims.size() == 3)
        {
            profile_batchnorm_forward_impl<BF16, BF16, F32, BF16, BF16, F32, 4, 3>(
                arg_parser.do_verification,
                arg_parser.init_method,
                arg_parser.do_dumpout,
                arg_parser.time_kernel,
                arg_parser.inLengths,
                arg_parser.reduceDims,
                arg_parser.updateMovingAverage,
                arg_parser.saveMeanAndInvVariance,
                epsilon,
                averageFactor);
        };
    }
    else if(arg_parser.data_type == 6)
    {
        if(arg_parser.inLengths.size() == 4 && arg_parser.reduceDims.size() == 3)
        {
            profile_batchnorm_forward_impl<F64, F64, F64, F64, F64, F64, 4, 3>(
                arg_parser.do_verification,
                arg_parser.init_method,
                arg_parser.do_dumpout,
                arg_parser.time_kernel,
                arg_parser.inLengths,
                arg_parser.reduceDims,
                arg_parser.updateMovingAverage,
                arg_parser.saveMeanAndInvVariance,
                epsilon,
                averageFactor);
        };
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("bnorm_fwd", "Batchnorm forward", profile_batchnorm_forward);
