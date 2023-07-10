// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <getopt.h>

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_reduce_impl.hpp"

using namespace ck;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"reduceDimensions", required_argument, nullptr, 'R'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths;
    std::vector<int> reduceDims;
    std::vector<float> scales;

    int data_type;
    int init_method = 1;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths "
                     "(only 4-d tensor supported)"
                  << std::endl;
        std::cout << "--reduceDimensions or -R comma seperated list of dimension indexes to reduce "
                     "(only 1 or 3 or 4 dimensions supported)"
                  << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "Arg1 -- data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2 -- init method(0=no init, 1=single integer value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:S:", long_options, &option_index);
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
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);
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

        data_type   = std::atoi(argv[optind++]);
        init_method = std::atoi(argv[optind]);

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        if(inLengths.size() != 4 ||
           (reduceDims.size() != 1 && reduceDims.size() != 3 && reduceDims.size() != 4))
            return (-1);

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5 && data_type != 6)
            return (-1);

        return (0);
    };
};

bool test_reduce_no_index(int data_type,
                          int init_method,
                          std::vector<int> reduceDims,
                          std::vector<size_t> inLengths,
                          ReduceTensorOp reduceOpId,
                          bool propagateNan,
                          float alpha,
                          float beta)
{
    using ck::profiler::profile_reduce_impl;

    bool result = true;

    if(data_type == 0)
    {
        result = profile_reduce_impl<float, float, float>(true,
                                                          init_method,
                                                          false,
                                                          false,
                                                          inLengths,
                                                          reduceDims,
                                                          reduceOpId,
                                                          propagateNan,
                                                          false,
                                                          alpha,
                                                          beta);
    }
    else if(data_type == 1)
    {
        result = profile_reduce_impl<ck::half_t, float, ck::half_t>(true,
                                                                    init_method,
                                                                    false,
                                                                    false,
                                                                    inLengths,
                                                                    reduceDims,
                                                                    reduceOpId,
                                                                    propagateNan,
                                                                    false,
                                                                    alpha,
                                                                    beta);
    }
    else if(data_type == 3)
    {
        result = profile_reduce_impl<int8_t, int32_t, int8_t>(true,
                                                              init_method,
                                                              false,
                                                              false,
                                                              inLengths,
                                                              reduceDims,
                                                              reduceOpId,
                                                              propagateNan,
                                                              false,
                                                              alpha,
                                                              beta);
    }
    else if(data_type == 5)
    {
        result = profile_reduce_impl<ck::bhalf_t, float, ck::bhalf_t>(true,
                                                                      init_method,
                                                                      false,
                                                                      false,
                                                                      inLengths,
                                                                      reduceDims,
                                                                      reduceOpId,
                                                                      propagateNan,
                                                                      false,
                                                                      alpha,
                                                                      beta);
    }
    else if(data_type == 6)
    {
        result = profile_reduce_impl<double, double, double>(true,
                                                             init_method,
                                                             false,
                                                             false,
                                                             inLengths,
                                                             reduceDims,
                                                             reduceOpId,
                                                             propagateNan,
                                                             false,
                                                             alpha,
                                                             beta);
    }

    return (result);
};

constexpr ReduceTensorOp reduceOpId = ReduceTensorOp::AVG;
constexpr bool propagateNan         = false;

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    bool result = true;

    if(argc == 1)
    {
        int data_type   = 1;
        int init_method = 2;
        std::vector<size_t> inLengths{64, 4, 280, 80};
        std::vector<std::vector<int>> v_reduceDims{
            {0, 1, 2, 3}, {0, 1, 2}, {1, 2, 3}, {0, 1, 3}, {0, 2, 3}, {0}, {1}, {2}, {3}};

        for(auto& reduceDims : v_reduceDims)
            result = result && test_reduce_no_index(data_type,
                                                    init_method,
                                                    reduceDims,
                                                    inLengths,
                                                    reduceOpId,
                                                    propagateNan,
                                                    1.0f,
                                                    0.0f);
    }
    else
    {
        if(args.processArgs(argc, argv) < 0)
        {
            throw std::runtime_error(
                "Invalid input arguments, test_reduce_no_index could not be executed!");
        };

        result = test_reduce_no_index(args.data_type,
                                      args.init_method,
                                      args.reduceDims,
                                      args.inLengths,
                                      reduceOpId,
                                      propagateNan,
                                      args.scales[0],
                                      args.scales[1]);
    }

    std::cout << "test_reduce_no_index ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
