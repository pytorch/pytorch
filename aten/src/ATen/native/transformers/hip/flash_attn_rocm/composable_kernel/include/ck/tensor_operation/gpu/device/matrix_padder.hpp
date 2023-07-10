// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename TensorDesc,
          typename TileLengths, // Tuple<...>
          typename DoPads>      // Sequence<bool, bool, ...>
__host__ __device__ constexpr auto
PadTensorDescriptor(const TensorDesc& desc, const TileLengths& tile_lengths, DoPads)
{
    constexpr index_t num_dim = DoPads::Size();

    static_assert(num_dim == TileLengths::Size() && num_dim == TensorDesc::GetNumOfDimension(),
                  "wrong! inconsistent # of dimensions");

    // transforms
    const auto transforms = generate_tuple(
        [&](auto idim) {
            const auto MRaw = desc.GetLength(idim);

            const auto MPerTile = tile_lengths[idim];

            const auto M = math::integer_divide_ceil(MRaw, MPerTile) * MPerTile;

            const auto MPad = M - MRaw;

            const bool DoPadM = DoPads::At(idim);

            const auto MTransform = conditional_expr<DoPadM>(make_right_pad_transform(MRaw, MPad),
                                                             make_pass_through_transform(MRaw));

            return MTransform;
        },
        Number<num_dim>{});

    // lower dimension Id
    const auto lower_dimss =
        generate_tuple([&](auto idim) { return Sequence<idim.value>{}; }, Number<num_dim>{});

    // upper dimension Id
    const auto upper_dimss = lower_dimss;

    return transform_tensor_descriptor(desc, transforms, lower_dimss, upper_dimss);
}

// M/N/K/OPerTileType could be index_t or Number<>
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType,
          typename OPerTileType>
struct GemmGemmPadder
{
    // TODO: hard to scale; use mask instead
    static constexpr bool PadM =
        GemmSpec == GemmSpecialization::MPadding || GemmSpec == GemmSpecialization::MNPadding ||
        GemmSpec == GemmSpecialization::MKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::MOPadding || GemmSpec == GemmSpecialization::MNOPadding ||
        GemmSpec == GemmSpecialization::MKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadN =
        GemmSpec == GemmSpecialization::NPadding || GemmSpec == GemmSpecialization::MNPadding ||
        GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::NOPadding || GemmSpec == GemmSpecialization::MNOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadK =
        GemmSpec == GemmSpecialization::KPadding || GemmSpec == GemmSpecialization::MKPadding ||
        GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding ||
        GemmSpec == GemmSpecialization::KOPadding || GemmSpec == GemmSpecialization::MKOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;
    static constexpr bool PadO =
        GemmSpec == GemmSpecialization::OPadding || GemmSpec == GemmSpecialization::MOPadding ||
        GemmSpec == GemmSpecialization::NOPadding || GemmSpec == GemmSpecialization::KOPadding ||
        GemmSpec == GemmSpecialization::MNOPadding || GemmSpec == GemmSpecialization::MKOPadding ||
        GemmSpec == GemmSpecialization::NKOPadding || GemmSpec == GemmSpecialization::MNKOPadding;

    // A[M, K]
    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor(
            a_desc_mraw_kraw, make_tuple(MPerTile_, KPerTile_), Sequence<PadM, PadK>{});
    }

    // B[K, N]
    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor(
            b_desc_nraw_kraw, make_tuple(NPerTile_, KPerTile_), Sequence<PadN, PadK>{});
    }

    // B1[Gemm1N, Gemm1K] = B1[O, N]
    template <typename B1Desc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadB1Descriptor_N_K(const B1Desc_NRaw_KRaw& b1_desc_nraw_kraw) const
    {
        return PadTensorDescriptor(
            b1_desc_nraw_kraw, make_tuple(OPerTile_, NPerTile_), Sequence<PadO, PadN>{});
    }

    // C[M, Gemm1N] = C[M, O]
    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor(
            c_desc_mraw_nraw, make_tuple(MPerTile_, OPerTile_), Sequence<PadM, PadO>{});
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
    OPerTileType OPerTile_;
};

// M/N/KPerTileType could be index_t or Number<>
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct GemmPadder
{
    static constexpr bool PadM =
        (GemmSpec == GemmSpecialization::MPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::MKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadN =
        (GemmSpec == GemmSpecialization::NPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadK =
        (GemmSpec == GemmSpecialization::KPadding || GemmSpec == GemmSpecialization::MKPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);

    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor(
            a_desc_mraw_kraw, make_tuple(MPerTile_, KPerTile_), Sequence<PadM, PadK>{});
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor(
            b_desc_nraw_kraw, make_tuple(NPerTile_, KPerTile_), Sequence<PadN, PadK>{});
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor(
            c_desc_mraw_nraw, make_tuple(MPerTile_, NPerTile_), Sequence<PadM, PadN>{});
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};

// Alias of GemmPadder; to deprecate
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct MatrixPadder : public GemmPadder<GemmSpec, MPerTileType, NPerTileType, KPerTileType>
{
};

// M/N/KPerTileType could be index_t or Number<>
template <bool PadM,
          bool PadN,
          bool PadK,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct GemmPadder_v2
{
    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor(
            a_desc_mraw_kraw, make_tuple(MPerTile_, KPerTile_), Sequence<PadM, PadK>{});
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor(
            b_desc_nraw_kraw, make_tuple(NPerTile_, KPerTile_), Sequence<PadN, PadK>{});
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor(
            c_desc_mraw_nraw, make_tuple(MPerTile_, NPerTile_), Sequence<PadM, PadN>{});
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};

// M/N/KPerTileType could be index_t or Number<>
template <bool PadM,
          bool PadN,
          bool PadK,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct MatrixPadder_v2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        const auto MRaw = a_desc_mraw_kraw.GetLength(I0);
        const auto KRaw = a_desc_mraw_kraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerTile_) * MPerTile_;
        const auto K = math::integer_divide_ceil(KRaw, KPerTile_) * KPerTile_;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(PadM && PadK)
        {
            // pad both M and K
            return transform_tensor_descriptor(a_desc_mraw_kraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(KRaw, KPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadM && (!PadK))
        {
            // pad M, but not K
            return transform_tensor_descriptor(
                a_desc_mraw_kraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(KRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadM) && PadK)
        {
            // pad K, but not M
            return transform_tensor_descriptor(
                a_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or K
            return a_desc_mraw_kraw;
        }
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        const auto NRaw = b_desc_nraw_kraw.GetLength(I0);
        const auto KRaw = b_desc_nraw_kraw.GetLength(I1);

        const auto N = math::integer_divide_ceil(NRaw, NPerTile_) * NPerTile_;
        const auto K = math::integer_divide_ceil(KRaw, KPerTile_) * KPerTile_;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(PadN && PadK)
        {
            // pad both N and K
            return transform_tensor_descriptor(b_desc_nraw_kraw,
                                               make_tuple(make_right_pad_transform(NRaw, NPad),
                                                          make_right_pad_transform(KRaw, KPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadN && (!PadK))
        {
            // pad N, but not K
            return transform_tensor_descriptor(
                b_desc_nraw_kraw,
                make_tuple(make_right_pad_transform(NRaw, NPad), make_pass_through_transform(KRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadN) && PadK)
        {
            // pad K, but not N
            return transform_tensor_descriptor(
                b_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad N or K
            return b_desc_nraw_kraw;
        }
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        const auto MRaw = c_desc_mraw_nraw.GetLength(I0);
        const auto NRaw = c_desc_mraw_nraw.GetLength(I1);

        const auto M = math::integer_divide_ceil(MRaw, MPerTile_) * MPerTile_;
        const auto N = math::integer_divide_ceil(NRaw, NPerTile_) * NPerTile_;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(PadM && PadN)
        {
            // pad M and N
            return transform_tensor_descriptor(c_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(PadM && (!PadN))
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr((!PadM) && PadN)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_desc_mraw_nraw;
        }
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
