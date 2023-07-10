// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/magic_division.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

__global__ void gpu_magic_number_division(uint32_t magic_multiplier,
                                          uint32_t magic_shift,
                                          const int32_t* p_dividend,
                                          int32_t* p_result,
                                          uint64_t num)
{
    uint64_t global_thread_num = blockDim.x * gridDim.x;

    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(uint64_t data_id = global_thread_id; data_id < num; data_id += global_thread_num)
    {
        p_result[data_id] =
            ck::MagicDivision::DoMagicDivision(p_dividend[data_id], magic_multiplier, magic_shift);
    }
}

__global__ void
gpu_naive_division(int32_t divisor, const int32_t* p_dividend, int32_t* p_result, uint64_t num)
{
    uint64_t global_thread_num = blockDim.x * gridDim.x;

    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(uint64_t data_id = global_thread_id; data_id < num; data_id += global_thread_num)
    {
        p_result[data_id] = p_dividend[data_id] / divisor;
    }
}

__host__ void cpu_magic_number_division(uint32_t magic_multiplier,
                                        uint32_t magic_shift,
                                        const int32_t* p_dividend,
                                        int32_t* p_result,
                                        uint64_t num)
{
    for(uint64_t data_id = 0; data_id < num; ++data_id)
    {
        p_result[data_id] =
            ck::MagicDivision::DoMagicDivision(p_dividend[data_id], magic_multiplier, magic_shift);
    }
}

int main(int, char*[])
{
    uint64_t num_divisor  = 4096;
    uint64_t num_dividend = 1L << 16;

    std::vector<int32_t> divisors_host(num_divisor);
    std::vector<int32_t> dividends_host(num_dividend);

    // generate divisor
    for(uint64_t i = 0; i < num_divisor; ++i)
    {
        divisors_host[i] = i + 1;
    }

    // generate dividend
    for(uint64_t i = 0; i < num_divisor; ++i)
    {
        dividends_host[i] = i;
    }

    DeviceMem dividends_dev_buf(sizeof(int32_t) * num_dividend);
    DeviceMem naive_result_dev_buf(sizeof(int32_t) * num_dividend);
    DeviceMem magic_result_dev_buf(sizeof(int32_t) * num_dividend);

    std::vector<int32_t> naive_result_host(num_dividend);
    std::vector<int32_t> magic_result_host(num_dividend);
    std::vector<int32_t> magic_result_host2(num_dividend);

    dividends_dev_buf.ToDevice(dividends_host.data());

    bool pass = true;

    for(std::size_t i = 0; i < num_divisor; ++i)
    {
        // run naive division on GPU
        gpu_naive_division<<<1024, 256>>>(
            divisors_host[i],
            static_cast<const int32_t*>(dividends_dev_buf.GetDeviceBuffer()),
            static_cast<int32_t*>(naive_result_dev_buf.GetDeviceBuffer()),
            num_dividend);

        // calculate magic number
        uint32_t magic_multiplier, magic_shift;

        ck::tie(magic_multiplier, magic_shift) =
            ck::MagicDivision::CalculateMagicNumbers(divisors_host[i]);

        // run magic division on GPU
        gpu_magic_number_division<<<1024, 256>>>(
            magic_multiplier,
            magic_shift,
            static_cast<const int32_t*>(dividends_dev_buf.GetDeviceBuffer()),
            static_cast<int32_t*>(magic_result_dev_buf.GetDeviceBuffer()),
            num_dividend);

        naive_result_dev_buf.FromDevice(naive_result_host.data());
        magic_result_dev_buf.FromDevice(magic_result_host.data());

        bool res = ck::utils::check_err(magic_result_host, naive_result_host);

        if(!res)
        {
            pass = false;
            continue;
        }

        cpu_magic_number_division(magic_multiplier,
                                  magic_shift,
                                  dividends_host.data(),
                                  magic_result_host2.data(),
                                  num_dividend);

        res = ck::utils::check_err(magic_result_host2, naive_result_host);

        if(!res)
        {
            pass = false;
            continue;
        }
    }

    if(pass)
    {
        std::cout << "test magic number division: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test magic number division: Fail" << std::endl;
        return -1;
    }
}
