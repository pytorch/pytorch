// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_STATIC_TENSOR_HPP
#define CK_STATIC_TENSOR_HPP

namespace ck {

// StaticTensor for Scalar
template <AddressSpaceEnum AddressSpace,
          typename T,
          typename TensorDesc,
          bool InvalidElementUseNumericalZeroValue,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
struct StaticTensor
{
    static constexpr auto desc_                  = TensorDesc{};
    static constexpr index_t ndim_               = TensorDesc::GetNumOfDimension();
    static constexpr index_t element_space_size_ = desc_.GetElementSpaceSize();

    __host__ __device__ constexpr StaticTensor() : invalid_element_scalar_value_{0} {}

    __host__ __device__ constexpr StaticTensor(T invalid_element_value)
        : invalid_element_scalar_value_{invalid_element_value}
    {
    }

    // read access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr const T& operator[](Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_[Number<offset>{}];
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return zero_scalar_value_;
            }
            else
            {
                return invalid_element_scalar_value_;
            }
        }
    }

    // write access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr T& operator()(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_(Number<offset>{});
        }
        else
        {
            return ignored_element_scalar_;
        }
    }

    StaticBuffer<AddressSpace, T, element_space_size_, true> data_;
    static constexpr T zero_scalar_value_ = T{0};
    const T invalid_element_scalar_value_;
    T ignored_element_scalar_;
};

// StaticTensor for vector
template <AddressSpaceEnum AddressSpace,
          typename S,
          index_t ScalarPerVector,
          typename TensorDesc,
          bool InvalidElementUseNumericalZeroValue,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
struct StaticTensorTupleOfVectorBuffer
{
    static constexpr auto desc_                  = TensorDesc{};
    static constexpr index_t ndim_               = TensorDesc::GetNumOfDimension();
    static constexpr index_t element_space_size_ = desc_.GetElementSpaceSize();

    static constexpr index_t num_of_vector_ =
        math::integer_divide_ceil(element_space_size_, ScalarPerVector);

    using V = vector_type<S, ScalarPerVector>;

    __host__ __device__ constexpr StaticTensorTupleOfVectorBuffer()
        : invalid_element_scalar_value_{0}
    {
    }

    __host__ __device__ constexpr StaticTensorTupleOfVectorBuffer(S invalid_element_value)
        : invalid_element_scalar_value_{invalid_element_value}
    {
    }

    // Get S
    // Idx is for S, not V
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr const S& operator[](Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_[Number<offset>{}];
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return zero_scalar_value_;
            }
            else
            {
                return invalid_element_scalar_value_;
            }
        }
    }

    // Set S
    // Idx is for S, not V
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr S& operator()(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_(Number<offset>{});
        }
        else
        {
            return ignored_element_scalar_;
        }
    }

    // Get X
    // Idx is for S, not X. Idx should be aligned with X
    template <typename X,
              typename Idx,
              typename enable_if<has_same_scalar_type<S, X>::value &&
                                     is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr X GetAsType(Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_.template GetAsType<X>(Number<offset>{});
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                // TODO: is this right way to initialize a vector?
                return X{0};
            }
            else
            {
                // TODO: is this right way to initialize a vector?
                return X{invalid_element_scalar_value_};
            }
        }
    }

    // Set X
    // Idx is for S, not X. Idx should be aligned with X
    template <typename X,
              typename Idx,
              typename enable_if<has_same_scalar_type<S, X>::value &&
                                     is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr void SetAsType(Idx, X x)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            data_.template SetAsType<X>(Number<offset>{}, x);
        }
    }

    // Get read access to V. No is_valid check
    // Idx is for S, not V. Idx should be aligned with V
    template <typename Idx>
    __host__ __device__ constexpr const V& GetVectorTypeReference(Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        return data_.GetVectorTypeReference(Number<offset>{});
    }

    // Get read access to V. No is_valid check
    // Idx is for S, not V. Idx should be aligned with V
    template <typename Idx>
    __host__ __device__ constexpr V& GetVectorTypeReference(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        return data_.GetVectorTypeReference(Number<offset>{});
    }

    StaticBufferTupleOfVector<AddressSpace, S, num_of_vector_, ScalarPerVector, true> data_;
    static constexpr S zero_scalar_value_ = S{0};
    const S invalid_element_scalar_value_ = S{0};
    S ignored_element_scalar_;
};

template <AddressSpaceEnum AddressSpace,
          typename T,
          typename TensorDesc,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
__host__ __device__ constexpr auto make_static_tensor(TensorDesc)
{
    return StaticTensor<AddressSpace, T, TensorDesc, true>{};
}

template <
    AddressSpaceEnum AddressSpace,
    typename T,
    typename TensorDesc,
    typename X,
    typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type                   = false,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto make_static_tensor(TensorDesc, X invalid_element_value)
{
    return StaticTensor<AddressSpace, T, TensorDesc, true>{invalid_element_value};
}

} // namespace ck
#endif
