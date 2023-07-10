// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
};

namespace gemm {

struct RowMajor : public BaseTensorLayout
{
    static constexpr const char* name = "RowMajor";
};

struct ColumnMajor : public BaseTensorLayout
{
    static constexpr const char* name = "ColumnMajor";
};
} // namespace gemm

namespace convolution {

// input tensor
// packed NCW/NCHW/NCDHW
struct NCW : public BaseTensorLayout
{
    static constexpr const char* name = "NCW";
};

struct NCHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCHW";
};

struct NCDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NCDHW";
};

// packed GNCW/GNCHW/GNCDHW
struct GNCW : public BaseTensorLayout
{
    static constexpr const char* name = "GNCW";
};

struct GNCHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNCHW";
};

struct GNCDHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNCDHW";
};

// input tensor
// packed NWC/NHWC/NDHWC
struct NWC : public BaseTensorLayout
{
    static constexpr const char* name = "NWC";
};

struct NHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NHWC";
};

struct NDHWC : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWC";
};

// input tensor
// packed GNWC/GNHWC/GNDHWC
struct GNWC : public BaseTensorLayout
{
    static constexpr const char* name = "GNWC";
};

struct GNHWC : public BaseTensorLayout
{
    static constexpr const char* name = "GNHWC";
};

struct GNDHWC : public BaseTensorLayout
{
    static constexpr const char* name = "GNDHWC";
};

// for input bias
struct GC : public BaseTensorLayout
{
    static constexpr const char* name = "GC";
};

// input tensor
// packed NWGC/NHWGC/NDHWGC
struct NWGC : public BaseTensorLayout
{
    static constexpr const char* name = "NWGC";
};

struct NHWGC : public BaseTensorLayout
{
    static constexpr const char* name = "NHWGC";
};

struct NDHWGC : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWGC";
};

// input tensor
// strided layout
struct G_NW_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_NW_C";
};

struct G_NHW_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_NHW_C";
};

struct G_NDHW_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_NDHW_C";
};

// for input bias
struct G_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_C";
};

// weight tensor
// packed KCX/KCYX/KCZYX
struct KCX : public BaseTensorLayout
{
    static constexpr const char* name = "KCX";
};

struct KCYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCYX";
};

struct KCZYX : public BaseTensorLayout
{
    static constexpr const char* name = "KCZYX";
};

// weight tensor
// packed KCX/KCYX/KCZYX
struct GKCX : public BaseTensorLayout
{
    static constexpr const char* name = "GKCX";
};

struct GKCYX : public BaseTensorLayout
{
    static constexpr const char* name = "GKCYX";
};

struct GKCZYX : public BaseTensorLayout
{
    static constexpr const char* name = "GKCZYX";
};

// weight tensor
// packed KXC/KYXC/KZYXC
struct KXC : public BaseTensorLayout
{
    static constexpr const char* name = "KXC";
};

struct KYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KYXC";
};

struct KZYXC : public BaseTensorLayout
{
    static constexpr const char* name = "KZYXC";
};

// weight tensor
// packed GKXC/GKYXC/GKZYXC
struct GKXC : public BaseTensorLayout
{
    static constexpr const char* name = "GKXC";
};

struct GKYXC : public BaseTensorLayout
{
    static constexpr const char* name = "GKYXC";
};

struct GKZYXC : public BaseTensorLayout
{
    static constexpr const char* name = "GKZYXC";
};

// weight tensor
// packed KXGC/KYXGC/KZYXGC
struct KXGC : public BaseTensorLayout
{
    static constexpr const char* name = "KXGC";
};

struct KYXGC : public BaseTensorLayout
{
    static constexpr const char* name = "KYXGC";
};

struct KZYXGC : public BaseTensorLayout
{
    static constexpr const char* name = "KZYXGC";
};

// weight tensor
// strided
struct G_K_X_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_K_X_C";
};

struct G_K_YX_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_K_YX_C";
};

struct G_K_ZYX_C : public BaseTensorLayout
{
    static constexpr const char* name = "G_K_ZYX_C";
};

// output tensor
// packed NKW/NKHW/NKDHW
struct NKW : public BaseTensorLayout
{
    static constexpr const char* name = "NKW";
};

struct NKHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKHW";
};

struct NKDHW : public BaseTensorLayout
{
    static constexpr const char* name = "NKDHW";
};

// output tensor
// packed GNKW/GNKHW/GNKDHW
struct GNKW : public BaseTensorLayout
{
    static constexpr const char* name = "GNKW";
};

struct GNKHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNKHW";
};

struct GNKDHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNKDHW";
};

// output tensor
// packed NWK/NHWK/NDHWK
struct NWK : public BaseTensorLayout
{
    static constexpr const char* name = "NWK";
};

struct NHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NHWK";
};

struct NDHWK : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWK";
};

// output tensor
// packed GNWK/GNHWK/GNDHWK
struct GNWK : public BaseTensorLayout
{
    static constexpr const char* name = "GNWK";
};

struct GNHWK : public BaseTensorLayout
{
    static constexpr const char* name = "GNHWK";
};

struct GNDHWK : public BaseTensorLayout
{
    static constexpr const char* name = "GNDHWK";
};

// for output bias
struct GK : public BaseTensorLayout
{
    static constexpr const char* name = "GK";
};

// output tensor
// packed NWGK/NHWGK/NDHWGK
struct NWGK : public BaseTensorLayout
{
    static constexpr const char* name = "NWGK";
};

struct NHWGK : public BaseTensorLayout
{
    static constexpr const char* name = "NHWGK";
};

struct NDHWGK : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWGK";
};

// output tensor
// strided layout
struct G_NW_K : public BaseTensorLayout
{
    static constexpr const char* name = "G_NW_K";
};

struct G_NHW_K : public BaseTensorLayout
{
    static constexpr const char* name = "G_NHW_K";
};

struct G_NDHW_K : public BaseTensorLayout
{
    static constexpr const char* name = "G_NDHW_K";
};

// for output bias
struct G_K : public BaseTensorLayout
{
    static constexpr const char* name = "G_K";
};

// K-reduced output tensor (packed)
struct GNW : public BaseTensorLayout
{
    static constexpr const char* name = "GNW";
};

struct GNHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNHW";
};

struct GNDHW : public BaseTensorLayout
{
    static constexpr const char* name = "GNDHW";
};

// K-reduced output tensor (packed)
struct NWG : public BaseTensorLayout
{
    static constexpr const char* name = "NWG";
};

struct NHWG : public BaseTensorLayout
{
    static constexpr const char* name = "NHWG";
};

struct NDHWG : public BaseTensorLayout
{
    static constexpr const char* name = "NDHWG";
};

// K-reduced output tensor (strided)
struct G_NW : public BaseTensorLayout
{
    static constexpr const char* name = "G_NW";
};

struct G_NHW : public BaseTensorLayout
{
    static constexpr const char* name = "G_NHW";
};

struct G_NDHW : public BaseTensorLayout
{
    static constexpr const char* name = "G_NDHW";
};

} // namespace convolution

template <
    typename Layout,
    typename std::enable_if<std::is_base_of<BaseTensorLayout, Layout>::value, bool>::type = false>
std::ostream& operator<<(std::ostream& os, const Layout&)
{
    os << Layout::name;
    return os;
}

} // namespace tensor_layout
} // namespace ck
