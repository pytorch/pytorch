// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "ck/ck.hpp"

namespace ck {

namespace host_common {

template <typename T>
static inline void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        std::cout << "Write output to file " << fileName << std::endl;
    }
    else
    {
        std::cout << "Could not open file " << fileName << " for writing" << std::endl;
    }
};

template <typename T>
static inline T getSingleValueFromString(const std::string& valueStr)
{
    std::istringstream iss(valueStr);

    T val;

    iss >> val;

    return (val);
};

template <typename T>
static inline std::vector<T> getTypeValuesFromString(const char* cstr_values)
{
    std::string valuesStr(cstr_values);

    std::vector<T> values;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = valuesStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        const std::string sliceStr = valuesStr.substr(pos, new_pos - pos);

        T val = getSingleValueFromString<T>(sliceStr);

        values.push_back(val);

        pos     = new_pos + 1;
        new_pos = valuesStr.find(',', pos);
    };

    std::string sliceStr = valuesStr.substr(pos);
    T val                = getSingleValueFromString<T>(sliceStr);

    values.push_back(val);

    return (values);
}

template <int NDim>
static inline std::vector<std::array<index_t, NDim>>
get_index_set(const std::array<index_t, NDim>& dim_lengths)
{
    static_assert(NDim >= 1, "NDim >= 1 is required to use this function!");

    if constexpr(NDim == 1)
    {
        std::vector<std::array<index_t, NDim>> index_set;

        for(int i = 0; i < dim_lengths[0]; i++)
        {
            std::array<index_t, 1> index{i};

            index_set.push_back(index);
        };

        return index_set;
    }
    else
    {
        std::vector<std::array<index_t, NDim>> index_set;
        std::array<index_t, NDim - 1> partial_dim_lengths;

        std::copy(dim_lengths.begin() + 1, dim_lengths.end(), partial_dim_lengths.begin());

        std::vector<std::array<index_t, NDim - 1>> partial_index_set;

        partial_index_set = get_index_set<NDim - 1>(partial_dim_lengths);

        for(index_t i = 0; i < dim_lengths[0]; i++)
            for(const auto& partial_index : partial_index_set)
            {
                std::array<index_t, NDim> index;

                index[0] = i;

                std::copy(partial_index.begin(), partial_index.end(), index.begin() + 1);

                index_set.push_back(index);
            };

        return index_set;
    };
};

template <int NDim>
static inline size_t get_offset_from_index(const std::array<index_t, NDim>& strides,
                                           const std::array<index_t, NDim>& index)
{
    size_t offset = 0;

    for(int i = 0; i < NDim; i++)
        offset += index[i] * strides[i];

    return (offset);
};

} // namespace host_common
} // namespace ck
