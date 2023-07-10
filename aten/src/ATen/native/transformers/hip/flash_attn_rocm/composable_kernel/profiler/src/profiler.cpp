// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>

#include "profiler_operation_registry.hpp"

static void print_helper_message()
{
    std::cout << "arg1: tensor operation " << ProfilerOperationRegistry::GetInstance() << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc == 1)
    {
        print_helper_message();
    }
    else if(const auto operation = ProfilerOperationRegistry::GetInstance().Get(argv[1]);
            operation.has_value())
    {
        return (*operation)(argc, argv);
    }
    else
    {
        std::cerr << "cannot find operation: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
}
