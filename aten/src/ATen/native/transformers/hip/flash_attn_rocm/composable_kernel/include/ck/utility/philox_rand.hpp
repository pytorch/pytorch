// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

class philox
{
    public:
    __device__ inline philox(unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset)
        : h_seed(reinterpret_cast<const uint2&>(seed))
    {

        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x    = offset / 4;
        tmp->y    = subsequence;
    }

    __device__ inline uint4 get_philox_4x32()
    {

        uint4 counter_ = counter;
        uint2 key_     = h_seed;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = single_loop(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = single_loop(counter_, key_);
        incr();

        return output;
    }

    __device__ inline uint4 get_philox_4x32(const unsigned long long subsequence)
    {

        uint4 counter_ = counter;
        ull2* tmp      = reinterpret_cast<ull2*>(&counter_);
        tmp->y         = subsequence;

        uint2 key_ = h_seed;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = single_loop(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = single_loop(counter_, key_);
        return output;
    }

    __device__ void get_random_8x16(ushort* out)
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32();

        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);

        out_tmp[0] = tmp_ph.x;
        out_tmp[1] = tmp_ph.y;
        out_tmp[2] = tmp_ph.z;
        out_tmp[3] = tmp_ph.w;
    }

    __device__ void get_random_8x16(ushort* out, const unsigned long long subsequence)
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);

        out_tmp[0] = tmp_ph.x;
        out_tmp[1] = tmp_ph.y;
        out_tmp[2] = tmp_ph.z;
        out_tmp[3] = tmp_ph.w;
    }

    private:
    struct ull2
    {
        uint64_t x;
        uint64_t y;
    };
    uint4 counter;
    const uint2 h_seed;

    __device__ uint4 incr(uint4 ctr)
    {

        uint4 res;
        res.x = ctr.x + 1;
        res.y = ctr.y;
        res.z = ctr.z;
        res.w = ctr.w;
        return res;
    }

    __device__ inline void incr() { counter = incr(counter); }

    __device__ uint2 u32_high_low_multi(const unsigned int a, const unsigned int b)
    {
        uint2* res;
        unsigned long long tmp;
        tmp = static_cast<unsigned long long>(a) * b;
        res = reinterpret_cast<uint2*>(&tmp);
        return *res;
    }

    __device__ inline uint4 single_loop(const uint4 ctr, const uint2 i_key)
    {

        uint2 res0 = u32_high_low_multi(kPhiloxSA, ctr.x);
        uint2 res1 = u32_high_low_multi(kPhiloxSB, ctr.z);
        uint4 ret  = {res1.y ^ ctr.y ^ i_key.x, res1.x, res0.y ^ ctr.w ^ i_key.y, res0.x};
        return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA  = 0xD2511F53;
    static const unsigned long kPhiloxSB  = 0xCD9E8D57;
};

} // namespace ck
