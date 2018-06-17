/*
    Copyright (c) 2005-2018 Intel Corporation

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

#include "tbb/blocked_range2d.h"
#include "harness_assert.h"

// First test as much as we can without including other headers.
// Doing so should catch problems arising from failing to include headers.

template<typename Tag>
class AbstractValueType {
    AbstractValueType() {}
    int value;
public:
    template<typename OtherTag>
    friend AbstractValueType<OtherTag> MakeAbstractValueType( int i );

    template<typename OtherTag>
    friend int GetValueOf( const AbstractValueType<OtherTag>& v ) ;
};

template<typename Tag>
AbstractValueType<Tag> MakeAbstractValueType( int i ) {
    AbstractValueType<Tag> x;
    x.value = i;
    return x;
}

template<typename Tag>
int GetValueOf( const AbstractValueType<Tag>& v ) {return v.value;}

template<typename Tag>
bool operator<( const AbstractValueType<Tag>& u, const AbstractValueType<Tag>& v ) {
    return GetValueOf(u)<GetValueOf(v);
}

template<typename Tag>
std::size_t operator-( const AbstractValueType<Tag>& u, const AbstractValueType<Tag>& v ) {
    return GetValueOf(u)-GetValueOf(v);
}

template<typename Tag>
AbstractValueType<Tag> operator+( const AbstractValueType<Tag>& u, std::size_t offset ) {
    return MakeAbstractValueType<Tag>(GetValueOf(u)+int(offset));
}

struct RowTag {};
struct ColTag {};

static void SerialTest() {
    typedef AbstractValueType<RowTag> row_type;
    typedef AbstractValueType<ColTag> col_type;
    typedef tbb::blocked_range2d<row_type,col_type> range_type;
    for( int row_x=-10; row_x<10; ++row_x ) {
        for( int row_y=row_x; row_y<10; ++row_y ) {
            row_type row_i = MakeAbstractValueType<RowTag>(row_x);
            row_type row_j = MakeAbstractValueType<RowTag>(row_y);
            for( int row_grain=1; row_grain<10; ++row_grain ) {
                for( int col_x=-10; col_x<10; ++col_x ) {
                    for( int col_y=col_x; col_y<10; ++col_y ) {
                        col_type col_i = MakeAbstractValueType<ColTag>(col_x);
                        col_type col_j = MakeAbstractValueType<ColTag>(col_y);
                        for( int col_grain=1; col_grain<10; ++col_grain ) {
                            range_type r( row_i, row_j, row_grain, col_i, col_j, col_grain );
                            AssertSameType( r.is_divisible(), true );
                            AssertSameType( r.empty(), true );
                            AssertSameType( static_cast<range_type::row_range_type::const_iterator*>(0), static_cast<row_type*>(0) );
                            AssertSameType( static_cast<range_type::col_range_type::const_iterator*>(0), static_cast<col_type*>(0) );
                            AssertSameType( r.rows(), tbb::blocked_range<row_type>( row_i, row_j, 1 ));
                            AssertSameType( r.cols(), tbb::blocked_range<col_type>( col_i, col_j, 1 ));
                            ASSERT( r.empty()==(row_x==row_y||col_x==col_y), NULL );
                            ASSERT( r.is_divisible()==(row_y-row_x>row_grain||col_y-col_x>col_grain), NULL );
                            if( r.is_divisible() ) {
                                range_type r2(r,tbb::split());
                                if( GetValueOf(r2.rows().begin())==GetValueOf(r.rows().begin()) ) {
                                    ASSERT( GetValueOf(r2.rows().end())==GetValueOf(r.rows().end()), NULL );
                                    ASSERT( GetValueOf(r2.cols().begin())==GetValueOf(r.cols().end()), NULL );
                                } else {
                                    ASSERT( GetValueOf(r2.cols().end())==GetValueOf(r.cols().end()), NULL );
                                    ASSERT( GetValueOf(r2.rows().begin())==GetValueOf(r.rows().end()), NULL );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#include "tbb/parallel_for.h"
#include "harness.h"

const int N = 1<<10;

unsigned char Array[N][N];

struct Striker {
   // Note: we use <int> here instead of <long> in order to test for problems similar to Quad 407676
    void operator()( const tbb::blocked_range2d<int>& r ) const {
        for( tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i )
            for( tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j )
                ++Array[i][j];
    }
};

void ParallelTest() {
    for( int i=0; i<N; i=i<3 ? i+1 : i*3 ) {
        for( int j=0; j<N; j=j<3 ? j+1 : j*3 ) {
            const tbb::blocked_range2d<int> r( 0, i, 7, 0, j, 5 );
            tbb::parallel_for( r, Striker() );
            for( int k=0; k<N; ++k ) {
                for( int l=0; l<N; ++l ) {
                    ASSERT( Array[k][l]==(k<i && l<j), NULL );
                    Array[k][l] = 0;
                }
            }
        }
    }
}

#include "tbb/task_scheduler_init.h"

int TestMain () {
    SerialTest();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        ParallelTest();
    }
    return Harness::Done;
}
