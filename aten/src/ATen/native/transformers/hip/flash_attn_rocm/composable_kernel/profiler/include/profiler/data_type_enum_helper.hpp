// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma

#include "ck/utility/data_type.hpp"
#include "profiler/data_type_enum.hpp"

namespace ck {

template <DataTypeEnum DataTypeEnum>
struct get_datatype_from_enum;

template <>
struct get_datatype_from_enum<DataTypeEnum::Int8>
{
    using type = int8_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum::Int32>
{
    using type = int32_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum::Half>
{
    using type = half_t;
};

template <>
struct get_datatype_from_enum<DataTypeEnum::Float>
{
    using type = float;
};

template <>
struct get_datatype_from_enum<DataTypeEnum::Double>
{
    using type = double;
};

template <typename T>
struct get_datatype_enum_from_type;

template <>
struct get_datatype_enum_from_type<int8_t>
{
    static constexpr DataTypeEnum value = DataTypeEnum::Int8;
};

template <>
struct get_datatype_enum_from_type<int32_t>
{
    static constexpr DataTypeEnum value = DataTypeEnum::Int32;
};

template <>
struct get_datatype_enum_from_type<half_t>
{
    static constexpr DataTypeEnum value = DataTypeEnum::Half;
};

template <>
struct get_datatype_enum_from_type<float>
{
    static constexpr DataTypeEnum value = DataTypeEnum::Float;
};

template <>
struct get_datatype_enum_from_type<double>
{
    static constexpr DataTypeEnum value = DataTypeEnum::Double;
};

} // namespace ck
