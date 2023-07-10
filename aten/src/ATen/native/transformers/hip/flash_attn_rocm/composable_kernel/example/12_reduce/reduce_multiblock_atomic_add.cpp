// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/utility/reduction_enums.hpp"
#include "reduce_multiblock_atomic_add_impl.hpp"
#include "reduce_example_common.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths = {16, 64, 32, 960};
    std::vector<int> reduceDims   = {0, 1, 2};
    std::vector<float> scales     = {1.0f, 0.0f};

    bool do_verification = true;
    int data_type        = 1;
    int init_method      = 2;
    bool time_kernel     = true;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--reduceDims or -R, comma separated list of to-reduce dimensions"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "Arg1: data type (0: fp32, 1: fp64)" << std::endl;
        std::cout << "Arg2 -- init method (0=no init, 1=single integer value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg3 -- time kernel (0=no, 1=yes)" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:v:l:", long_options, &option_index);
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

        if(optind + 3 > argc)
        {
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");
        };

        data_type   = std::atoi(argv[optind++]);
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

template <typename InOutDataType,
          typename AccDataType,
          ReduceTensorOp ReduceOpId,
          index_t PropagateNan>
bool reduce_multiblock_atomic_add_test(bool do_verification,
                                       int init_method,
                                       bool time_kernel,
                                       const std::vector<size_t>& inLengths,
                                       const std::vector<int>& reduceDims,
                                       float alpha,
                                       float beta)
{
    bool matched = false;
    int result   = 0;

    const auto tuple_object = reduce_shape_instances{};

    static_for<0, std::tuple_size<reduce_shape_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using ShapeType = remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(ShapeType::Rank_ != inLengths.size() || ShapeType::NumReduceDim_ != reduceDims.size())
            return;

        std::array<int, ShapeType::NumReduceDim_> a_reduceDims;

        ck::ranges::copy(reduceDims, a_reduceDims.begin());

        result = reduce_multiblock_atomic_add_impl<InOutDataType,
                                                   AccDataType,
                                                   ReduceOpId,
                                                   ShapeType::Rank_,
                                                   ShapeType::NumReduceDim_,
                                                   PropagateNan>(
            do_verification, init_method, time_kernel, inLengths, a_reduceDims, alpha, beta);

        matched = true;
    });

    return (result == 0) ? true : false;
};

constexpr ReduceTensorOp ReduceOpId = ReduceTensorOp::AVG;
constexpr bool PropagateNan         = true;

int main(int argc, char* argv[])
{
    bool pass = true;

    if(argc > 1)
    {
        SimpleAppArgs arg;

        if(arg.processArgs(argc, argv) < 0)
            return (-1);

        if(arg.data_type == 0)
        {
            pass = reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                arg.do_verification,
                arg.init_method,
                arg.time_kernel,
                arg.inLengths,
                arg.reduceDims,
                arg.scales[0],
                arg.scales[1]);
        }
        else if(arg.data_type == 1)
        {
            pass = reduce_multiblock_atomic_add_test<double, double, ReduceOpId, PropagateNan>(
                arg.do_verification,
                arg.init_method,
                arg.time_kernel,
                arg.inLengths,
                arg.reduceDims,
                arg.scales[0],
                arg.scales[1]);
        }
    }
    else
    {
        // for testing float
        pass = pass && reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, 2, false, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing double
        pass = pass && reduce_multiblock_atomic_add_test<double, double, ReduceOpId, PropagateNan>(
                           true, 2, false, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing 3D input
        pass = pass && reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, 2, false, {16, 64, 960}, {0, 1}, 1.0f, 0.0f);

        // for testing 5D input
        pass = pass && reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, 2, false, {16, 64, 32, 2, 960}, {0, 1, 2, 3}, 1.0f, 0.0f);
    };

    return (pass ? 0 : 1);
};
