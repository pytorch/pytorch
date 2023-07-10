// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/data_type_enum.hpp"
#include "profiler/profile_groupnorm_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

struct GroupnormArgParser
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

#define OP_NAME "groupnorm"
#define OP_DESC "Group Normalization"

void print_help_groupnorm()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: fp16; 1: fp32)\n"
              << "arg3: verification (0: no; 1: yes)\n"
              << "arg4: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg5: print tensor value (0: no; 1: yes)\n"
              << "arg6: time kernel (0=no, 1=yes)\n"
              << "--length: tensor extents (e.g, --length 1 16 16 32 40) \n"
              << std::endl;
}

int profile_groupnorm(int argc, char* argv[])
{
    ck::DataTypeEnum data_type  = ck::DataTypeEnum::Half;
    bool do_verification        = false;
    int init_method             = 0;
    bool do_log                 = 0;
    bool time_kernel            = 1;
    std::vector<index_t> length = {64, 16, 16, 32, 40};

    if(argc != 1 && argc != 13)
    {
        print_help_groupnorm();
        return 0;
    }

    if(argc == 13)
    {
        data_type       = static_cast<ck::DataTypeEnum>(std::stoi(argv[2]));
        do_verification = std::stoi(argv[3]);
        init_method     = std::stoi(argv[4]);
        do_log          = std::stoi(argv[5]);
        time_kernel     = std::stoi(argv[6]);

        // parse the long options
        GroupnormArgParser arg_parser;
        arg_parser(argc, argv);
        length = arg_parser.long_opts["length"];
    }

    using F16 = ck::half_t;
    using F32 = float;

    if(data_type == ck::DataTypeEnum::Float)
    {
        ck::profiler::profile_groupnorm_impl<F32, F32, F32, F32, F32>(
            do_verification, init_method, do_log, time_kernel, length);
    }
    else if(data_type == ck::DataTypeEnum::Half)
    {
        ck::profiler::profile_groupnorm_impl<F16, F16, F16, F32, F16>(
            do_verification, init_method, do_log, time_kernel, length);
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_groupnorm);
