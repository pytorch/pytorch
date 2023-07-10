// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_base.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim_m,
          index_t NumDim_n,
          index_t MPerThread,
          index_t NPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DeviceElementwise : public DeviceElementwiseBase<InDataTypeTuple,
                                                        OutDataTypeTuple,
                                                        ElementwiseOperation,
                                                        NumDim_m + NumDim_n>
{
    static constexpr index_t NumDim = NumDim_m + NumDim_n;

    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    };

    static auto GenerateOutDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

                return static_cast<DataType*>(nullptr);
            },
            Number<NumOutput>{});
    };

    using InDataTypePointerTuple  = decltype(GenerateInDataTypePointerTuple());
    using OutDataTypePointerTuple = decltype(GenerateOutDataTypePointerTuple());

    template <typename Desc_MN>
    static auto PadDescriptor_MN_2d(Desc_MN desc_mn,
                                    index_t gridSize,
                                    index_t blockSize,
                                    index_t num_threads_m,
                                    index_t num_threads_n)
    {
        std::ignore               = blockSize;
        std::ignore               = gridSize;
        const auto m              = desc_mn.GetLength(I0);
        const auto n              = desc_mn.GetLength(I1);
        const index_t loop_step_m = num_threads_m * MPerThread;
        const index_t loop_step_n = num_threads_n * NPerThread;
        const auto pad_m          = math::integer_least_multiple(m, loop_step_m) - m;
        const auto pad_n          = math::integer_least_multiple(n, loop_step_n) - n;

        const auto desc_mn_pad = transform_tensor_descriptor(
            desc_mn,
            make_tuple(make_right_pad_transform(m, pad_m), make_right_pad_transform(n, pad_n)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        return desc_mn_pad;
    }

    static auto MakeDescriptor_MN(const std::array<index_t, NumDim>& lengths,
                                  const std::array<index_t, NumDim>& stride,
                                  index_t gridSize,
                                  index_t blockSize,
                                  index_t num_threads_m,
                                  index_t num_threads_n)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NumDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<NumDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDim_m, 1>::type();
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDim_m, NumDim_m + NumDim_n, 1>::type();

        const auto mLengths = get_container_subset(tupleOfShape, mDimIds);
        const auto nLengths = get_container_subset(tupleOfShape, nDimIds);

        // merge nd to 2d desc - [s0 * s1 * ...]

        if constexpr(NumDim > 2)
        {
            const auto desc_mn = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
                make_tuple(mDimIds, nDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return PadDescriptor_MN_2d(desc_mn, gridSize, blockSize, num_threads_m, num_threads_n);
        }
        else
            return PadDescriptor_MN_2d(desc, gridSize, blockSize, num_threads_m, num_threads_n);
    }

    template <index_t TupleSize>
    static auto GenerateInOutGrid2dDescTuple(Number<TupleSize>)
    {
        return generate_tuple(
            [&](auto) {
                if constexpr(NumDim > 2)
                {
                    return MakeDescriptor_MN({1, 1}, {1, 1}, 1, 1, 1, 1);
                }
                else
                {
                    return MakeDescriptor_MN({1}, {1}, 1, 1, 1, 1);
                };
            },
            Number<TupleSize>{});
    };

    using OutGrid2dDescTuple = decltype(GenerateInOutGrid2dDescTuple(Number<NumOutput>{}));
    using InGrid2dDescTuple  = decltype(GenerateInOutGrid2dDescTuple(Number<NumInput>{}));

    using GridwiseElementwise = GridwiseElementwise_2D<InGrid2dDescTuple,
                                                       OutGrid2dDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       ElementwiseOperation,
                                                       MPerThread,
                                                       NPerThread,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op)

            : lengths_(lengths),
              inStridesArray_(inStridesArray),
              outStridesArray_(outStridesArray),
              elementwise_op_(elementwise_op),
              blockSize_(256),
              gridSize_(120), // FIXME - Calculate the grid size by number of CU in the future
              num_threads_m_((gridSize_ * blockSize_) / 16),
              num_threads_n_(16)
        {
            static_assert(NumDim_m > 0, "");
            static_assert(NumDim_n > 0, "");

            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            out_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;
                    return static_cast<DataType*>(out_dev_buffers[I.value]);
                },
                Number<NumOutput>{});

            in_grid_2d_desc_tuple_ = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_MN(lengths,
                                             inStridesArray[I.value],
                                             gridSize_,
                                             blockSize_,
                                             num_threads_m_,
                                             num_threads_n_);
                },
                Number<NumInput>{});

            out_grid_2d_desc_tuple_ = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_MN(lengths,
                                             outStridesArray[I.value],
                                             gridSize_,
                                             blockSize_,
                                             num_threads_m_,
                                             num_threads_n_);
                },
                Number<NumOutput>{});
        }

        InDataTypePointerTuple in_dev_buffers_;
        OutDataTypePointerTuple out_dev_buffers_;
        InGrid2dDescTuple in_grid_2d_desc_tuple_;
        OutGrid2dDescTuple out_grid_2d_desc_tuple_;

        std::array<index_t, NumDim> lengths_;
        std::array<std::array<index_t, NumDim>, NumInput> inStridesArray_;
        std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray_;

        ElementwiseOperation elementwise_op_;
        index_t blockSize_;
        index_t gridSize_;
        index_t num_threads_m_;
        index_t num_threads_n_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_elementwise_2d<GridwiseElementwise,
                                                      InGrid2dDescTuple,
                                                      OutGrid2dDescTuple,
                                                      InDataTypePointerTuple,
                                                      OutDataTypePointerTuple,
                                                      ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.in_grid_2d_desc_tuple_,
                                                        arg.out_grid_2d_desc_tuple_,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        arg.elementwise_op_,
                                                        arg.num_threads_m_,
                                                        arg.num_threads_n_);
            return elapsed_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg == nullptr)
            return false;

        if(pArg->lengths_.back() % MPerThread != 0)
            return false;

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector,
                                          index_t vectorDim) {
            if(strides[vectorDim] == 1 &&
               (lengths[vectorDim] % scalarPerVector == 0 ||
                lengths[vectorDim] % scalarPerVector == lengths[vectorDim]))
            {
                return true;
            }
            if(strides[vectorDim] != 1 && scalarPerVector == strides[vectorDim])
            {
                return true;
            }
            return false;
        };

        bool valid = true;
        static_for<0, NumInput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(pArg->lengths_,
                                       pArg->inStridesArray_[I.value],
                                       InScalarPerVectorSeq::At(I),
                                       NumDim_m - 1))
                valid = false;
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(pArg->lengths_,
                                       pArg->outStridesArray_[I.value],
                                       OutScalarPerVectorSeq::At(I),
                                       NumDim - 1))
                valid = false;
        });

        return valid;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          outStridesArray,
                                          in_dev_buffers,
                                          out_dev_buffers,
                                          elementwise_op);
    }

    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
