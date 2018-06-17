/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#define TBB_PREVIEW_BLOCKED_RANGE_ND 1
#include "tbb/blocked_rangeNd.h"

#include "tbb/tbb_config.h"

#if __TBB_CPP11_PRESENT && __TBB_CPP11_ARRAY_PRESENT && __TBB_CPP11_TEMPLATE_ALIASES_PRESENT
// AbstractValueType class represents Value concept's requirements in the most abstract way
class AbstractValueType {
    int value;
    AbstractValueType() {}
public:
    friend AbstractValueType MakeAbstractValue(int i);
    friend int GetValueOf(const AbstractValueType& v);
};

int GetValueOf(const AbstractValueType& v) { return v.value; }

AbstractValueType MakeAbstractValue(int i) {
    AbstractValueType x;
    x.value = i;
    return x;
}

// operator- returns amount of elements of AbstractValueType between u and v
std::size_t operator-(const AbstractValueType& u, const AbstractValueType& v) {
    return GetValueOf(u) - GetValueOf(v);
}

bool operator<(const AbstractValueType& u, const AbstractValueType& v) {
    return GetValueOf(u) < GetValueOf(v);
}

AbstractValueType operator+(const AbstractValueType& u, std::size_t offset) {
    return MakeAbstractValue(GetValueOf(u) + int(offset));
}

#include "harness_assert.h"
#include <algorithm> // std::for_each
#include <array>

namespace internal {
    template<typename range_t, unsigned int N>
    struct utils {
        using val_t = typename range_t::value_type;

        template<typename EntityType, std::size_t DimSize>
        using data_type = std::array<typename utils<range_t, N - 1>::template data_type<EntityType, DimSize>, DimSize>;

        template<typename EntityType, std::size_t DimSize>
        static void init_data(data_type<EntityType, DimSize>& data) {
            std::for_each(data.begin(), data.end(), utils<range_t, N - 1>::template init_data<EntityType, DimSize>);
        }

        template<typename EntityType, std::size_t DimSize>
        static void increment_data(const range_t& range, data_type<EntityType, DimSize>& data) {
            auto begin = data.begin() + range.dim(N - 1).begin();
            // same as "auto end = out.begin() + range.dim(N - 1).end();"
            auto end = begin + range.dim(N - 1).size();
            for (auto i = begin; i != end; ++i) {
                utils<range_t, N - 1>::template increment_data<EntityType, DimSize>(range, *i);
            }
        }

        template<typename EntityType, std::size_t DimSize>
        static void check_data(const range_t& range, data_type<EntityType, DimSize>& data) {
            auto begin = data.begin() + range.dim(N - 1).begin();
            // same as "auto end = out.begin() + range.dim(N - 1).end();"
            auto end = begin + range.dim(N - 1).size();
            for (auto i = begin; i != end; ++i) {
                utils<range_t, N - 1>::template check_data<EntityType, DimSize>(range, *i);
            }
        }

        template<typename input_t, std::size_t... Is>
        static range_t make_range(std::size_t shift, bool negative, val_t(*gen)(input_t), tbb::internal::index_sequence<Is...>) {
            return range_t( { {
                    /*    begin =*/gen(negative ? -input_t(Is + shift) : 0),
                    /*      end =*/gen(input_t(Is + shift)),
                    /*grainsize =*/Is + 1}
                    /*pack expansion*/... } );
        }

        static bool is_empty(const range_t& range) {
            if (range.dim(N - 1).empty()) { return true; }
            return utils<range_t, N - 1>::is_empty(range);
        }

        static bool is_divisible(const range_t& range) {
            if (range.dim(N - 1).is_divisible()) { return true; }
            return utils<range_t, N - 1>::is_divisible(range);
        }

        static void check_splitting(const range_t& range_split, const range_t& range_new, int(*get)(const val_t&), bool split_checker = false) {
            if (get(range_split.dim(N - 1).begin()) == get(range_new.dim(N - 1).begin())) {
                ASSERT(get(range_split.dim(N - 1).end()) == get(range_new.dim(N - 1).end()), NULL);
            }
            else {
                ASSERT(get(range_split.dim(N - 1).end()) == get(range_new.dim(N - 1).begin()) && !split_checker, NULL);
                split_checker = true;
            }
            utils<range_t, N - 1>::check_splitting(range_split, range_new, get, split_checker);
        }

    };

    template<typename range_t>
    struct utils<range_t, 0> {
        using val_t = typename range_t::value_type;

        template<typename EntityType, std::size_t DimSize>
        using data_type = EntityType;

        template<typename EntityType, std::size_t DimSize>
        static void init_data(data_type<EntityType, DimSize>& data) { data = 0; }

        template<typename EntityType, std::size_t DimSize>
        static void increment_data(const range_t&, data_type<EntityType, DimSize>& data) { ++data; }

        template<typename EntityType, std::size_t DimSize>
        static void check_data(const range_t&, data_type<EntityType, DimSize>& data) {
            ASSERT(data == 1, NULL);
        }

        static bool is_empty(const range_t&) { return false; }

        static bool is_divisible(const range_t&) { return false; }

        static void check_splitting(const range_t&, const range_t&, int(*)(const val_t&), bool) {}
    };

    // We need MakeInt function to pass it into make_range as factory function
    // because of matching make_range with AbstractValueType and other types too
    int MakeInt(int i) { return i; }
}

template<unsigned int DimAmount>
void SerialTest() {
    __TBB_STATIC_ASSERT((tbb::blocked_rangeNd<int, DimAmount>::ndims()
                         == tbb::blocked_rangeNd<AbstractValueType, DimAmount>::ndims()),
                         "different amount of dimensions");

    using range_t = tbb::blocked_rangeNd<AbstractValueType, DimAmount>;
    // 'typedef' instead of 'using' because of GCC 4.7.2 bug on Debian 7.0
    typedef internal::utils<range_t, DimAmount> utils;

    // Generate empty range
    range_t r = utils::make_range(0, true, &MakeAbstractValue, tbb::internal::make_index_sequence<DimAmount>());

    AssertSameType(r.is_divisible(), bool());
    AssertSameType(r.empty(), bool());
    AssertSameType(range_t::ndims(), 0U);

    ASSERT(r.empty() == utils::is_empty(r) && r.empty(), NULL);
    ASSERT(r.is_divisible() == utils::is_divisible(r), NULL);

    // Generate not-empty range divisible range
    r = utils::make_range(1, true, &MakeAbstractValue, tbb::internal::make_index_sequence<DimAmount>());
    ASSERT(r.empty() == utils::is_empty(r) && !r.empty(), NULL);
    ASSERT(r.is_divisible() == utils::is_divisible(r) && r.is_divisible(), NULL);

    range_t r_new(r, tbb::split());
    utils::check_splitting(r, r_new, &GetValueOf);

    SerialTest<DimAmount - 1>();
}
template<> void SerialTest<0>() {}

#include "tbb/parallel_for.h"

template<unsigned int DimAmount>
void ParallelTest() {
    using range_t = tbb::blocked_rangeNd<int, DimAmount>;
    // 'typedef' instead of 'using' because of GCC 4.7.2 bug on Debian 7.0
    typedef internal::utils<range_t, DimAmount>  utils;

    // Max size is                                 1 << 20 - 1 bytes
    // Thus size of one dimension's elements is    1 << (20 / DimAmount - 1) bytes
    typename utils::template data_type<unsigned char, 1 << (20 / DimAmount - 1)> data;
    utils::init_data(data);

    range_t r = utils::make_range((1 << (20 / DimAmount - 1)) - DimAmount, false, &internal::MakeInt, tbb::internal::make_index_sequence<DimAmount>());

    tbb::parallel_for(r, [&data](const range_t& range) {
        utils::increment_data(range, data);
    });

    utils::check_data(r, data);

    ParallelTest<DimAmount - 1>();
}
template<> void ParallelTest<0>() {}

void TestCtors() {
    tbb::blocked_rangeNd<int, 1>{ { 0,13,3 } };

    tbb::blocked_rangeNd<int, 1>{ tbb::blocked_range<int>{ 0,13,3 } };

    tbb::blocked_rangeNd<int, 2>(tbb::blocked_range<int>(-8923, 8884, 13), tbb::blocked_range<int>(-8923, 5, 13));

    tbb::blocked_rangeNd<int, 2>({ -8923, 8884, 13 }, { -8923, 8884, 13 });

    tbb::blocked_range<int> r1(0, 13);

    tbb::blocked_range<int> r2(-12, 23);

    tbb::blocked_rangeNd<int, 2>({ { -8923, 8884, 13 }, r1});

    tbb::blocked_rangeNd<int, 2>({ r2, r1 });

    tbb::blocked_rangeNd<int, 2>(r1, r2);

    tbb::blocked_rangeNd<AbstractValueType, 4>({ MakeAbstractValue(-3), MakeAbstractValue(13), 8 },
                                               { MakeAbstractValue(-53), MakeAbstractValue(23), 2 },
                                               { MakeAbstractValue(-23), MakeAbstractValue(33), 1 },
                                               { MakeAbstractValue(-13), MakeAbstractValue(43), 7 });
}

static const std::size_t N = 4;

#include "harness.h"
#include "tbb/task_scheduler_init.h"

int TestMain() {
    TestCtors();
    SerialTest<N>();
    for( int p=MinThread; p<= MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        ParallelTest<N>();
    }
    return Harness::Done;
}

#else

// tbb::blocked_rangeNd requires C++11 support
#define HARNESS_SKIP_TEST 1
#include "harness.h"

#endif /* __TBB_CPP11_PRESENT && __TBB_CPP11_ARRAY_PRESENT && __TBB_CPP11_TEMPLATE_ALIASES_PRESENT */
