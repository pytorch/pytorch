// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/data_type_enum.hpp"
#include "profiler/profile_layernorm_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

struct LayernormArgParser
{
    std::unordered_map<std::string, std::vector<int>> long_opts = {{"length", {}}};

    bool parse_opt(int argc, char* argv[], const std::string& key, int i)
    {
        if(std::string("--") + key == argv[i])
        {
            int pos = i;
            while(++i < argc && argv[i][0] != '-') {}
            int end = i;
            for(int j = pos + 1; j < end; j++)
            {
                long_opts[key].push_back(std::stoi(argv[j]));
            }
            return true;
        }
        return false;
    }

    void operator()(int argc, char* argv[])
    {
        for(auto& kv : long_opts)
        {
            for(int i = 1; i < argc; i++)
            {
                if(parse_opt(argc, argv, kv.first, i))
                    break;
            }
        }
    }
};

void print_help_layernorm()
{
    std::cout << "arg1: data type (0: fp16; 1: fp32)\n"
              << "arg2: verification (0: no; 1: yes)\n"
              << "arg3: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg4: print tensor value (0: no; 1: yes)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "--length: tensor extents (e.g, --length 1024 1024) \n"
              << std::endl;
}

int profile_layernorm(int argc, char* argv[])
{
    if(argc <= 2)
    {
        print_help_layernorm();
        return 0;
    }

    LayernormArgParser arg_parser;

    // short unnamed options
    const ck::DataTypeEnum data_type = static_cast<ck::DataTypeEnum>(std::stoi(argv[2]));
    const bool do_verification       = std::stoi(argv[3]);
    const int init_method            = std::stoi(argv[4]);
    const bool do_log                = std::stoi(argv[5]);
    const bool time_kernel           = std::stoi(argv[6]);

    // parse the long options
    arg_parser(argc, argv);
    const std::vector<index_t> length = arg_parser.long_opts["length"];

    using F16          = ck::half_t;
    using F32          = float;
    constexpr int rank = 2;

    if(data_type == ck::DataTypeEnum::Half)
    {
        ck::profiler::profile_layernorm_impl<F16, F16, F16, F32, F16, rank>(
            do_verification, init_method, do_log, time_kernel, length);
    }
    else if(data_type == ck::DataTypeEnum::Float)
    {
        ck::profiler::profile_layernorm_impl<F32, F32, F32, F32, F32, rank>(
            do_verification, init_method, do_log, time_kernel, length);
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("layernorm", "Layer Normalization", profile_layernorm);
