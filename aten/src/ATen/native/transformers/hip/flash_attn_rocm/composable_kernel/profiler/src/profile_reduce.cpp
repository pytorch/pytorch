// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <getopt.h>

#include "ck/utility/reduction_enums.hpp"

#include "ck/library/utility/host_common_util.hpp"

#include "profiler/profile_reduce_impl.hpp"
#include "profiler/data_type_enum.hpp"
#include "profiler_operation_registry.hpp"

using namespace std;

using ck::ReduceTensorOp;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"reduceDims", required_argument, nullptr, 'R'},
                                       {"reduceOp", required_argument, nullptr, 'O'},
                                       {"compType", required_argument, nullptr, 'C'},
                                       {"outType", required_argument, nullptr, 'W'},
                                       {"nanOpt", required_argument, nullptr, 'N'},
                                       {"indicesOpt", required_argument, nullptr, 'I'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"double", no_argument, nullptr, '?'},
                                       {"int8", no_argument, nullptr, '?'},
                                       {"bf16", no_argument, nullptr, '?'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

static void check_reduce_dims(const int rank, const std::vector<int>& reduceDims)
{
    for(auto dim : reduceDims)
    {
        if(dim < 0 || dim >= rank)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };

    unsigned int flag = 0;

    for(auto dim : reduceDims)
    {
        if(flag & (0x1 << dim))
            throw std::runtime_error("All toReduce dimensions should be different!");
        flag = flag | (0x1 << dim);
    };
};

class ReduceProfilerArgs
{
    private:
    int option_index = 0;

    public:
    bool use_half   = false;
    bool use_double = false;
    bool use_int8   = false;
    bool use_bf16   = false;

    std::vector<size_t> inLengths;
    std::vector<size_t> outLengths;
    std::vector<int> reduceDims;

    std::vector<float> scales;

    ReduceTensorOp reduceOp     = ReduceTensorOp::ADD;
    ck::DataTypeEnum compTypeId = ck::DataTypeEnum::Float;
    ck::DataTypeEnum outTypeId  = ck::DataTypeEnum::Float;

    bool compType_assigned = false;
    bool outType_assigned  = false;

    int nanOpt           = 0;
    int indicesOpt       = 0;
    bool do_verification = false;
    bool do_dumpout      = false;

    int init_method;
    bool time_kernel;

    ReduceProfilerArgs()  = default;
    ~ReduceProfilerArgs() = default;

    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--reduceDims or -R, comma separated list of to-reduce dimensions"
                  << std::endl;
        std::cout << "--reduceOp or -O, enum value indicating the reduction operations"
                  << std::endl;
        std::cout << "--compType or -C, enum value indicating the type of accumulated values used "
                     "during the reduction"
                  << std::endl;
        std::cout << "--outType or -W, optional enum value indicating the type of the reduced "
                     "output, which could be float when the input data is half"
                  << std::endl;
        std::cout
            << "--nanOpt or -N, 1/0 value indicates the selection to use or not use Nan-Propagation"
            << std::endl;
        std::cout << "--indicesOpt or -I, 1/0 value indicates the selection to use or not use "
                     "index in reduction"
                  << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
        std::cout << "--double, use fp64 for the input and output tensor data types" << std::endl;
        std::cout << "--int8, use int8 for the input and output tensor data types" << std::endl;
        std::cout << "--bf16, use bfloat16 for the input and output tensor data types" << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "--dumpout or -o, 1/0 to indicate where to save the reduction result to files "
                     "for further analysis"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        optind++; // to skip the "reduce" module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:O:C:W:N:I:S:v:o:", long_options, &option_index);
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
            case 'O':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                reduceOp = static_cast<ReduceTensorOp>(std::atoi(optarg));
                break;
            case 'C':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                compTypeId        = static_cast<ck::DataTypeEnum>(std::atoi(optarg));
                compType_assigned = true;
                break;
            case 'W':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                outTypeId        = static_cast<ck::DataTypeEnum>(std::atoi(optarg));
                outType_assigned = true;
                break;
            case 'N':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                nanOpt = std::atoi(optarg);
                break;
            case 'I':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                indicesOpt = std::atoi(optarg);
                break;
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);

                if(scales.size() != 2)
                    throw std::runtime_error("Invalid option format!");
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
                if(std::string(long_options[option_index].name) == "half")
                    use_half = true;
                else if(std::string(long_options[option_index].name) == "double")
                    use_double = true;
                else if(std::string(long_options[option_index].name) == "int8")
                    use_int8 = true;
                else if(std::string(long_options[option_index].name) == "bf16")
                    use_bf16 = true;
                else if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;

            default:
                show_usage(argv[0]);
                std::cerr << "Invalid cmd-line options!" << std::endl;
                return (-1);
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

        if(reduceOp == ReduceTensorOp::MIN || reduceOp == ReduceTensorOp::MAX ||
           reduceOp == ReduceTensorOp::AMAX)
        {
            // for indexable operations, no need to assign compType and outType, just let them be
            // same as inType
            compType_assigned = false;
            outType_assigned  = false;
        };

        return (0);
    };

}; // end of class AppArgs

int profile_reduce(int argc, char* argv[])
{
    using ck::DataTypeEnum;
    using ck::profiler::profile_reduce_impl;

    ReduceProfilerArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inLengths.size();

    check_reduce_dims(rank, args.reduceDims);

    if(args.reduceOp == ReduceTensorOp::MUL || args.reduceOp == ReduceTensorOp::NORM1)
        throw std::runtime_error("MUL and NORM1 are not supported by composable kernel!");

    if(args.use_half)
    {
        if(!args.compType_assigned)
            args.compTypeId = DataTypeEnum::Half;

        if(args.outType_assigned &&
           (args.outTypeId != DataTypeEnum::Half && args.outTypeId != DataTypeEnum::Float))
            args.outTypeId = DataTypeEnum::Float;

        if(!args.outType_assigned)
            args.outTypeId = DataTypeEnum::Half;

        if(args.compTypeId == DataTypeEnum::Half)
        {
            profile_reduce_impl<ck::half_t, ck::half_t, ck::half_t>(
                args.do_verification,
                args.init_method,
                args.do_dumpout,
                args.time_kernel,
                args.inLengths,
                args.reduceDims,
                args.reduceOp,
                static_cast<bool>(args.nanOpt),
                static_cast<bool>(args.indicesOpt),
                args.scales[0],
                args.scales[1]);
        }
        else if(args.compTypeId == DataTypeEnum::Float)
        {
            profile_reduce_impl<ck::half_t, float, ck::half_t>(args.do_verification,
                                                               args.init_method,
                                                               args.do_dumpout,
                                                               args.time_kernel,
                                                               args.inLengths,
                                                               args.reduceDims,
                                                               args.reduceOp,
                                                               static_cast<bool>(args.nanOpt),
                                                               static_cast<bool>(args.indicesOpt),
                                                               args.scales[0],
                                                               args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(args.use_double)
    {
        profile_reduce_impl<double, double, double>(args.do_verification,
                                                    args.init_method,
                                                    args.do_dumpout,
                                                    args.time_kernel,
                                                    args.inLengths,
                                                    args.reduceDims,
                                                    args.reduceOp,
                                                    static_cast<bool>(args.nanOpt),
                                                    static_cast<bool>(args.indicesOpt),
                                                    args.scales[0],
                                                    args.scales[1]);
    }
    else if(args.use_int8)
    {
        if(!args.compType_assigned)
            args.compTypeId = DataTypeEnum::Int8;

        if(args.outType_assigned &&
           (args.outTypeId != DataTypeEnum::Int8 && args.outTypeId != DataTypeEnum::Int32))
            args.outTypeId = DataTypeEnum::Int32;

        if(!args.outType_assigned)
            args.outTypeId = DataTypeEnum::Int8;

        if(args.compTypeId == DataTypeEnum::Int8)
        {
            profile_reduce_impl<int8_t, int8_t, int8_t>(args.do_verification,
                                                        args.init_method,
                                                        args.do_dumpout,
                                                        args.time_kernel,
                                                        args.inLengths,
                                                        args.reduceDims,
                                                        args.reduceOp,
                                                        static_cast<bool>(args.nanOpt),
                                                        static_cast<bool>(args.indicesOpt),
                                                        args.scales[0],
                                                        args.scales[1]);
        }
        else if(args.compTypeId == DataTypeEnum::Int32)
        {
            profile_reduce_impl<int8_t, int32_t, int8_t>(args.do_verification,
                                                         args.init_method,
                                                         args.do_dumpout,
                                                         args.time_kernel,
                                                         args.inLengths,
                                                         args.reduceDims,
                                                         args.reduceOp,
                                                         static_cast<bool>(args.nanOpt),
                                                         static_cast<bool>(args.indicesOpt),
                                                         args.scales[0],
                                                         args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(args.use_bf16)
    {
        if(args.outType_assigned &&
           (args.outTypeId != DataTypeEnum::BFloat16 && args.outTypeId != DataTypeEnum::Float))
            args.outTypeId = DataTypeEnum::Float;

        if(!args.outType_assigned)
            args.outTypeId = DataTypeEnum::BFloat16;

        profile_reduce_impl<ck::bhalf_t, float, ck::bhalf_t>(args.do_verification,
                                                             args.init_method,
                                                             args.do_dumpout,
                                                             args.time_kernel,
                                                             args.inLengths,
                                                             args.reduceDims,
                                                             args.reduceOp,
                                                             static_cast<bool>(args.nanOpt),
                                                             static_cast<bool>(args.indicesOpt),
                                                             args.scales[0],
                                                             args.scales[1]);
    }
    else
    {
        if(args.compTypeId == DataTypeEnum::Float)
        {
            profile_reduce_impl<float, float, float>(args.do_verification,
                                                     args.init_method,
                                                     args.do_dumpout,
                                                     args.time_kernel,
                                                     args.inLengths,
                                                     args.reduceDims,
                                                     args.reduceOp,
                                                     static_cast<bool>(args.nanOpt),
                                                     static_cast<bool>(args.indicesOpt),
                                                     args.scales[0],
                                                     args.scales[1]);
        }
        else if(args.compTypeId == DataTypeEnum::Double)
        {
            profile_reduce_impl<float, double, float>(args.do_verification,
                                                      args.init_method,
                                                      args.do_dumpout,
                                                      args.time_kernel,
                                                      args.inLengths,
                                                      args.reduceDims,
                                                      args.reduceOp,
                                                      static_cast<bool>(args.nanOpt),
                                                      static_cast<bool>(args.indicesOpt),
                                                      args.scales[0],
                                                      args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    };

    return (0);
};

REGISTER_PROFILER_OPERATION("reduce", "Reduce", profile_reduce);
