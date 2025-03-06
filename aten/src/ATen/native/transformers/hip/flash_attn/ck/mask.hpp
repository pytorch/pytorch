// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>

// keep this in sync with ck_tile::GenericAttentionMaskEnum
enum class mask_enum
{
    no_mask = 0,
    mask_top_left,
    mask_bottom_right,
    window_generic,
};

struct mask_info
{
    mask_enum type;
    ck_tile::index_t y, x;
    ck_tile::index_t left, right; // FA style SWA left/right

    void serialize(std::ostream& os) const
    {
        if(type == mask_enum::no_mask)
            os << "n";
        else if(type == mask_enum::mask_top_left)
            os << "t(" << left << ":" << right << ")";
        else if(type == mask_enum::mask_bottom_right)
            os << "b(" << left << ":" << right << ")";
        else
        {
            os << "g(" << y << ":" << x << ")";
        }
    }
    static mask_info decode(std::string str, ck_tile::index_t seqlen_q, ck_tile::index_t seqlen_k)
    {
        ck_tile::index_t x_total = seqlen_k;
        ck_tile::index_t y_total = seqlen_q;
        mask_info tmp;
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            if(t == "xt" || t == "xb")
            {
                // xformer style sliding window attn from top-left
                ck_tile::index_t window_size = atoi(v.c_str());
                ck_tile::index_t left_size   = -1;
                ck_tile::index_t right_size  = 0;
                if(window_size > 0)
                {
                    left_size  = window_size / 2;
                    right_size = window_size - 1 - left_size;
                }
                auto r = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                    left_size, right_size, y_total, x_total, t == "xt");

                tmp.type  = t == "xt" ? mask_enum::mask_top_left : mask_enum::mask_bottom_right;
                tmp.y     = r.at(ck_tile::number<0>{});
                tmp.x     = r.at(ck_tile::number<1>{});
                tmp.left  = left_size;
                tmp.right = right_size;
            }
            else
            {
                auto found_1 = v.find(",");
                if(found_1 == std::string::npos)
                {
                    printf("not supported value %s, %s\n", v.c_str(), str.c_str());
                    assert(0);
                }
                tmp.type            = mask_enum::window_generic;
                ck_tile::index_t v0 = atoi(v.substr(0, found_1).c_str());
                ck_tile::index_t v1 = atoi(v.substr(found_1 + 1).c_str());
                // TODO: some validation
                if(t == "t")
                {
                    tmp.type = mask_enum::mask_top_left;
                    auto r   = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                        v0, v1, y_total, x_total, true);
                    tmp.y     = r.at(ck_tile::number<0>{});
                    tmp.x     = r.at(ck_tile::number<1>{});
                    tmp.left  = v0;
                    tmp.right = v1;
                }
                else if(t == "b")
                {
                    tmp.type = mask_enum::mask_bottom_right;
                    auto r   = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                        v0, v1, y_total, x_total, false);
                    tmp.y     = r.at(ck_tile::number<0>{});
                    tmp.x     = r.at(ck_tile::number<1>{});
                    tmp.left  = v0;
                    tmp.right = v1;
                }
                else if(t == "g")
                {
                    tmp.y     = v0;
                    tmp.x     = v1;
                    tmp.left  = v0; // TODO: don't use this?
                    tmp.right = v1;
                }
                else
                {
                    printf("not supported type %s, %s\n", t.c_str(), str.c_str());
                    assert(0);
                }
            }
        }
        else
        {
            auto set_causal_top_left = [&]() {
                tmp.type  = mask_enum::mask_top_left;
                tmp.y     = seqlen_q;
                tmp.x     = 1;
                tmp.left  = -1;
                tmp.right = 0;
            };
            auto set_causal_bottom_right = [&]() {
                tmp.type  = mask_enum::mask_bottom_right;
                tmp.y     = seqlen_q;
                tmp.x     = seqlen_k - seqlen_q + 1;
                tmp.left  = -1;
                tmp.right = 0;
            };
            if(str == "t")
                set_causal_top_left();
            else if(str == "b")
                set_causal_bottom_right();
            else
            {
                tmp.type = static_cast<mask_enum>(atoi(str.c_str()));
                if(tmp.type == mask_enum::mask_top_left)
                {
                    set_causal_top_left();
                }
                else if(tmp.type == mask_enum::mask_bottom_right)
                {
                    set_causal_bottom_right();
                }
            }
        }
        return tmp;
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi)
    {
        mi.serialize(os);
        return os;
    }
};
