// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ck/utility/data_type.hpp"
#include "ck/utility/span.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/ranges.hpp"

template <typename Range>
std::ostream& LogRange(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << v;
    }
    return os;
}

template <typename T, typename Range>
std::ostream& LogRangeAsType(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << static_cast<T>(v);
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct HostTensorDescriptor
{
    HostTensorDescriptor() = default;

    void CalculateStrides();

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                          std::is_convertible_v<Y, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t> &&
                  std::is_convertible_v<ck::ranges::range_value_t<Strides>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens, const Strides& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    std::size_t GetNumOfDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpaceSize() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->GetNumOfDimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    std::size_t GetOffsetFromMultiIndex(std::vector<std::size_t> iss) const
    {
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

template <typename New2Old>
HostTensorDescriptor transpose_host_tensor_descriptor_given_new2old(const HostTensorDescriptor& a,
                                                                    const New2Old& new2old)
{
    std::vector<std::size_t> new_lengths(a.GetNumOfDimension());
    std::vector<std::size_t> new_strides(a.GetNumOfDimension());

    for(std::size_t i = 0; i < a.GetNumOfDimension(); i++)
    {
        new_lengths[i] = a.GetLengths()[new2old[i]];
        new_strides[i] = a.GetStrides()[new2old[i]];
    }

    return HostTensorDescriptor(new_lengths, new_strides);
}

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&) = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [=] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(mF, GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct Tensor
{
    using Descriptor = HostTensorDescriptor;
    using Data       = std::vector<T>;

    template <typename X>
    Tensor(std::initializer_list<X> lens) : mDesc(lens), mData(mDesc.GetElementSpaceSize())
    {
    }

    template <typename X, typename Y>
    Tensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : mDesc(lens, strides), mData(mDesc.GetElementSpaceSize())
    {
    }

    template <typename Lengths>
    Tensor(const Lengths& lens) : mDesc(lens), mData(mDesc.GetElementSpaceSize())
    {
    }

    template <typename Lengths, typename Strides>
    Tensor(const Lengths& lens, const Strides& strides)
        : mDesc(lens, strides), mData(GetElementSpaceSize())
    {
    }

    Tensor(const Descriptor& desc) : mDesc(desc), mData(mDesc.GetElementSpaceSize()) {}

    template <typename OutT>
    Tensor<OutT> CopyAsType() const
    {
        Tensor<OutT> ret(mDesc);

        ck::ranges::transform(
            mData, ret.mData.begin(), [](auto value) { return ck::type_convert<OutT>(value); });

        return ret;
    }

    Tensor()              = delete;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&)      = default;

    ~Tensor() = default;

    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    template <typename FromT>
    explicit Tensor(const Tensor<FromT>& other) : Tensor(other.template CopyAsType<T>())
    {
    }

    decltype(auto) GetLengths() const { return mDesc.GetLengths(); }

    decltype(auto) GetStrides() const { return mDesc.GetStrides(); }

    std::size_t GetNumOfDimension() const { return mDesc.GetNumOfDimension(); }

    std::size_t GetElementSize() const { return mDesc.GetElementSize(); }

    std::size_t GetElementSpaceSize() const { return mDesc.GetElementSpaceSize(); }

    std::size_t GetElementSpaceSizeInBytes() const { return sizeof(T) * GetElementSpaceSize(); }

    void SetZero() { ck::ranges::fill<T>(mData, 0); }

    template <typename F>
    void ForEach_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(F&& f)
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void ForEach_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(const F&& f) const
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.GetNumOfDimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0], mDesc.GetLengths()[1])(num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(
                f, mDesc.GetLengths()[0], mDesc.GetLengths()[1], mDesc.GetLengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4])(num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4],
                                       mDesc.GetLengths()[5])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...)];
    }

    T& operator()(std::vector<std::size_t> idx)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx)];
    }

    const T& operator()(std::vector<std::size_t> idx) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx)];
    }

    Tensor<T> Transpose(std::vector<size_t> axes = {}) const
    {
        if(axes.empty())
        {
            axes.resize(this->GetNumOfDimension());
            std::iota(axes.rbegin(), axes.rend(), 0);
        }
        if(axes.size() != mDesc.GetNumOfDimension())
        {
            throw std::runtime_error(
                "Tensor::Transpose(): size of axes must match tensor dimension");
        }
        std::vector<size_t> tlengths, tstrides;
        for(const auto& axis : axes)
        {
            tlengths.push_back(GetLengths()[axis]);
            tstrides.push_back(GetStrides()[axis]);
        }
        Tensor<T> ret(*this);
        ret.mDesc = HostTensorDescriptor(tlengths, tstrides);
        return ret;
    }

    Tensor<T> Transpose(std::vector<size_t> axes = {})
    {
        return const_cast<Tensor<T> const*>(this)->Transpose(axes);
    }

    typename Data::iterator begin() { return mData.begin(); }

    typename Data::iterator end() { return mData.end(); }

    typename Data::pointer data() { return mData.data(); }

    typename Data::const_iterator begin() const { return mData.begin(); }

    typename Data::const_iterator end() const { return mData.end(); }

    typename Data::const_pointer data() const { return mData.data(); }

    typename Data::size_type size() const { return mData.size(); }

    template <typename U = T>
    auto AsSpan() const
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::add_const_t<std::remove_reference_t<U>>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    template <typename U = T>
    auto AsSpan()
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::remove_reference_t<U>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    Descriptor mDesc;
    Data mData;
};

template <typename T>
void SerializeTensor(std::ostream& os,
                     const Tensor<T>& tensor,
                     std::vector<size_t>& idx,
                     size_t rank)
{
    if(rank == tensor.mDesc.GetNumOfDimension() - 1)
    {
        os << "(";
        for(size_t i = 0; i < rank; i++)
        {
            os << idx[i] << (i == rank - 1 ? ", x) : " : ", ");
        }

        size_t dimz = tensor.mDesc.GetLengths()[rank];

        os << "[";
        for(size_t i = 0; i < dimz; i++)
        {
            idx[rank] = i;
            os << tensor(idx) << (i == dimz - 1 ? "]" : ", ");
        }
        os << "\n";
        return;
    }

    for(size_t i = 0; i < tensor.mDesc.GetLengths()[rank]; i++)
    {
        idx[rank] = i;
        SerializeTensor(os, tensor, idx, rank + 1);
    }
}

// Example format for Tensor(2, 2, 3):
// (0, 0, x) : [0, 1, 2]
// (0, 1, x) : [3, 4, 5]
// (1, 0, x) : [6, 7, 8]
// (1, 1, x) : [9, 10, 11]
template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    std::vector<size_t> idx(tensor.mDesc.GetNumOfDimension(), 0);
    SerializeTensor(os, tensor, idx, 0);
    return os;
}
