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

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_multiple_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "dual_reduce_common.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType       = ck::half_t;
using OutDataType      = float;
using OutDataTypeTuple = Tuple<OutDataType, OutDataType>;
using AccDataType      = float;

// for NHWC layer-norm calculation of mean and meansquare
constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

constexpr bool PropagateNan = false;

constexpr InMemoryDataOperationEnum OutMemoryDataOperation = InMemoryDataOperationEnum::Set;

using DeviceDualReduce = DeviceMultipleReduceMultiBlock<2,
                                                        InDataType,
                                                        AccDataType,
                                                        OutDataTypeTuple,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceOperation,
                                                        InElementwiseOperationTuple,
                                                        AccElementwiseOperationTuple,
                                                        OutMemoryDataOperation,
                                                        PropagateNan,
                                                        256,
                                                        4,
                                                        64,
                                                        1,
                                                        1,
                                                        1, // InSrcVectorDim
                                                        1,
                                                        ck::Sequence<1, 1>>;

int main(int argc, char* argv[])
{
    int retval = 0;

    if(argc > 1)
    {
        SimpleAppArgs arg;

        if(arg.processArgs(argc, argv) < 0)
            return (-1);

        std::array<int, NumReduceDim> reduceDims = {1, 2, 3};

        retval = mean_meansquare_dual_reduce_test<DeviceDualReduce,
                                                  InDataType,
                                                  OutDataType,
                                                  AccDataType,
                                                  Rank,
                                                  NumReduceDim>(arg.n,
                                                                arg.h,
                                                                arg.w,
                                                                arg.c,
                                                                arg.do_verification,
                                                                arg.init_method,
                                                                arg.time_kernel,
                                                                reduceDims);
    }
    else
    {
        std::array<int, NumReduceDim> reduceDims = {1, 2, 3};

        retval = mean_meansquare_dual_reduce_test<DeviceDualReduce,
                                                  InDataType,
                                                  OutDataType,
                                                  AccDataType,
                                                  Rank,
                                                  NumReduceDim>(
            600, 28, 28, 256, true, 2, true, reduceDims);
    };

    return (retval);
}
