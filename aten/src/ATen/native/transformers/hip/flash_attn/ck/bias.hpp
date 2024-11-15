// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

struct bias_info
{
    bias_enum type;
    /*
     * simple dispatch logic
     *
     * if type == elementwise_bias:
     *      if rank_info == 0:
     *           bias is 1*1*s*s
     *      elif rank_info == 1:
     *           bias is 1*h*s*s
     *      elif rank_info == 2:
     *           bias is b*h*s*s
     *
     * elif type == alibi:
     *       if rank_info == 0:
     *           alibi in 1*h
     *       elif rank_info == 1:
     *           alibi in b*h
     */
    int rank_info;

    void serialize(std::ostream& os) const
    {
        if(type == bias_enum::no_bias)
            os << "n";
        else if(type == bias_enum::elementwise_bias)
        {
            os << "e";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
        else if(type == bias_enum::alibi)
        {
            os << "alibi";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
    }

    static bias_info decode(std::string str)
    {
        bias_info info{bias_enum::no_bias, 0};
        if(str == "0" || str == "n")
        {
            info.type = bias_enum::no_bias;
        }
        else if(str.compare(0, 1, "1") == 0 || str.compare(0, 1, "e") == 0 ||
                str.compare(0, 11, "elementwise") == 0)
        {
            info.type    = bias_enum::elementwise_bias;
            auto found_0 = str.find(':');
            if(found_0 != std::string::npos)
            {
                std::string e  = str.substr(found_0 + 1);
                info.rank_info = atoi(e.c_str());
            }
        }
        else if(str.compare(0, 1, "2") == 0 || str.compare(0, 1, "a") == 0 ||
                str.compare(0, 5, "alibi") == 0)
        {
            info.type    = bias_enum::alibi;
            auto found_0 = str.find(':');
            if(found_0 != std::string::npos)
            {
                std::string e  = str.substr(found_0 + 1);
                info.rank_info = atoi(e.c_str());
            }
        }
        return info;
    }

    friend std::ostream& operator<<(std::ostream& os, const bias_info& bi)
    {
        bi.serialize(os);
        return os;
    }
};
