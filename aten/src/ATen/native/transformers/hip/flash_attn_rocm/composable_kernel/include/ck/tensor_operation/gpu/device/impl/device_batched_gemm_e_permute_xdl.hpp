#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_e_permute.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for
\link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemmCPermute and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename ABDataType,
          typename EDataType,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_e_permute_xdl(const ABDataType* __restrict__ p_a_grid,
                                          const ABDataType* __restrict__ p_b_grid,
                                          EDataType* __restrict__ p_e_grid,
                                          const index_t batch_count,
                                          const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
                                          const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
                                          const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                              e_grid_desc_mblock_mperblock_nblock_nperblock,
                                          const AElementwiseOperation a_element_op,
                                          const BElementwiseOperation b_element_op,
                                          const CDEElementwiseOperation cde_element_op,
                                          const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                          const Block2ETileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  ck::Tuple<>{},
                                                  p_e_grid + e_batch_offset,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ck::Tuple<>{},
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_etile_map;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumPrefetch,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceBatchedGemmEPermuteXdl : public DeviceBatchedGemmEPermute<ALayout,
                                                                       BLayout,
                                                                       ELayout,
                                                                       ADataType,
                                                                       BDataType,
                                                                       EDataType,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmEPermuteXdl;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    static auto MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    static auto
    MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t stride_M, index_t stride_N)
    {
        const auto e_grid_desc_mraw_nraw =
            make_naive_tensor_descriptor(make_tuple(MRaw, NRaw), make_tuple(stride_M, stride_N));

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto MakeEGridDescriptor_G0_G1_M_N(index_t G0,
                                              index_t G1,
                                              index_t MRaw,
                                              index_t NRaw,
                                              index_t stride_G0,
                                              index_t stride_G1,
                                              index_t stride_M,
                                              index_t stride_N)
    {
        const auto e_grid_desc_g0_g1_mraw_nraw = [&]() {
            return make_naive_tensor_descriptor(
                make_tuple(G0, G1, MRaw, NRaw),
                make_tuple(stride_G0, stride_G1, stride_M, stride_N));
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(
                e_grid_desc_g0_g1_mraw_nraw,
                make_tuple(make_pass_through_transform(G0),
                           make_pass_through_transform(G1),
                           make_right_pad_transform(MRaw, MPad),
                           make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                e_grid_desc_g0_g1_mraw_nraw,
                make_tuple(make_pass_through_transform(G0),
                           make_pass_through_transform(G1),
                           make_right_pad_transform(MRaw, MPad),
                           make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                e_grid_desc_g0_g1_mraw_nraw,
                make_tuple(make_pass_through_transform(G0),
                           make_pass_through_transform(G1),
                           make_pass_through_transform(MRaw),
                           make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
        }
        else
        {
            // not pad M or N
            return e_grid_desc_g0_g1_mraw_nraw;
        }
    }

    using AGridDesc_M_K       = decltype(MakeAGridDescriptor_M_K(1, 1, 1));
    using BGridDesc_N_K       = decltype(MakeBGridDescriptor_N_K(1, 1, 1));
    using EGridDesc_M_N       = decltype(MakeEGridDescriptor_M_N(1, 1, 1, 1));
    using EGridDesc_G0_G1_M_N = decltype(MakeEGridDescriptor_G0_G1_M_N(1, 1, 1, 1, 1, 1, 1, 1));

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t Batchstride_A,
                                       index_t Batchstride_B,
                                       EGridDesc_G0_G1_M_N e_grid_desc_g0_g1_m_n)
            : Batchstride_A_(Batchstride_A),
              Batchstride_B_(Batchstride_B),
              e_grid_desc_g0_g1_m_n_(e_grid_desc_g0_g1_m_n)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(Batchstride_A_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(Batchstride_B_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            const index_t G1 = e_grid_desc_g0_g1_m_n_.GetLength(I1);
            index_t b0       = g_idx / G1;
            index_t b1       = g_idx - b0 * G1; // g_idx % G1
            return e_grid_desc_g0_g1_m_n_.CalculateOffset(make_multi_index(b0, b1, 0, 0));
        }

        private:
        index_t Batchstride_A_;
        index_t Batchstride_B_;
        EGridDesc_G0_G1_M_N e_grid_desc_g0_g1_m_n_;
    };

    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>, // DsDataType,
        EDataType,   // EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_M_K,
        BGridDesc_N_K,
        Tuple<>,
        EGridDesc_M_N,
        NumPrefetch,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;

    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = decltype(
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}));
    using Block2ETileMap = typename GridwiseGemm::DefaultBlock2ETileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 EDataType* p_e_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t stride_A,
                 index_t stride_B,
                 index_t batch_stride_A,
                 index_t batch_stride_B,
                 BatchedGemmEPermuteDesc batched_gemm_e_permute_desc,
                 index_t BatchCount,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_e_grid_{p_e_grid},
              BatchCount_(BatchCount),
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K(M, K, stride_A)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K(K, N, stride_B)},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N(batched_gemm_e_permute_desc.M_,
                                                    batched_gemm_e_permute_desc.N_,
                                                    batched_gemm_e_permute_desc.stride_M_,
                                                    batched_gemm_e_permute_desc.stride_N_)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              e_grid_desc_mblock_mperblock_nblock_nperblock{},
              e_grid_desc_g0_g1_m_n_{
                  DeviceOp::MakeEGridDescriptor_G0_G1_M_N(batched_gemm_e_permute_desc.G0_,
                                                          batched_gemm_e_permute_desc.G1_,
                                                          batched_gemm_e_permute_desc.M_,
                                                          batched_gemm_e_permute_desc.N_,
                                                          batched_gemm_e_permute_desc.stride_G0_,
                                                          batched_gemm_e_permute_desc.stride_G1_,
                                                          batched_gemm_e_permute_desc.stride_M_,
                                                          batched_gemm_e_permute_desc.stride_N_)},
              compute_ptr_offset_of_batch_{batch_stride_A, batch_stride_B, e_grid_desc_g0_g1_m_n_},
              block_2_etile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_m_k_,
                                           b_grid_desc_n_k_,
                                           ck::Tuple<>{},
                                           e_grid_desc_m_n_,
                                           block_2_etile_map_))
            {
                e_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);
            }
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            std::cout << "C[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        // batch count
        index_t BatchCount_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock;
        EGridDesc_G0_G1_M_N e_grid_desc_g0_g1_m_n_;

        // for calculating Batch offset
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            ck::Tuple<>{},
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemmCPermute_km_kn_m0m1n0n1_xdlops_v2r3 has invalid "
                    "setting");
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * arg.BatchCount_;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_batched_gemm_e_permute_xdl<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    EDataType,
                    remove_reference_t<DeviceOp::AGridDesc_AK0_M_AK1>,
                    remove_reference_t<DeviceOp::BGridDesc_BK0_N_BK1>,
                    typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch,
                    remove_reference_t<Block2ETileMap>,
                    has_main_k_block_loop_>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_e_grid_,
                                              arg.BatchCount_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.compute_ptr_offset_of_batch_,
                                              arg.block_2_etile_map_);
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                           arg.b_grid_desc_n_k_,
                                           ck::Tuple<>{},
                                           arg.e_grid_desc_m_n_,
                                           arg.block_2_etile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             EDataType* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t stride_A,
                             index_t stride_B,
                             index_t batch_stride_A,
                             index_t batch_stride_B,
                             BatchedGemmEPermuteDesc batched_gemm_e_permute_desc,
                             index_t BatchCount,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_e,
                        M,
                        N,
                        K,
                        stride_A,
                        stride_B,
                        batch_stride_A,
                        batch_stride_B,
                        batched_gemm_e_permute_desc,
                        BatchCount,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_A,
                        index_t stride_B,
                        index_t batch_stride_A,
                        index_t batch_stride_B,
                        BatchedGemmEPermuteDesc batched_gemm_e_permute_desc,
                        index_t BatchCount,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<EDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          batch_stride_A,
                                          batch_stride_B,
                                          batched_gemm_e_permute_desc,
                                          BatchCount,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemmEPermuteXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
