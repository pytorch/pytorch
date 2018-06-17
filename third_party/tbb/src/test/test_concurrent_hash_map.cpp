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

#ifndef TBB_USE_PERFORMANCE_WARNINGS
#define TBB_USE_PERFORMANCE_WARNINGS 1
#endif

// Our tests usually include the header under test first.  But this test needs
// to use the preprocessor to edit the identifier runtime_warning in concurrent_hash_map.h.
// Hence we include a few other headers before doing the abusive edit.
#include "tbb/tbb_stddef.h" /* Defines runtime_warning */
#include "harness_assert.h" /* Prerequisite for defining hooked_warning */

// The symbol internal::runtime_warning is normally an entry point into the TBB library.
// Here for sake of testing, we define it to be hooked_warning, a routine peculiar to this unit test.
#define runtime_warning hooked_warning

static bool bad_hashing = false;

namespace tbb {
    namespace internal {
        static void hooked_warning( const char* /*format*/, ... ) {
            ASSERT(bad_hashing, "unexpected runtime_warning: bad hashing");
        }
    } // namespace internal
} // namespace tbb
#define __TBB_EXTRA_DEBUG 1 // enables additional checks
#include "tbb/concurrent_hash_map.h"

// Restore runtime_warning as an entry point into the TBB library.
#undef runtime_warning

namespace Jungle {
    struct Tiger {};
    size_t tbb_hasher( const Tiger& ) {return 0;}
}

#if !defined(_MSC_VER) || _MSC_VER>=1400 || __INTEL_COMPILER
void test_ADL() {
    tbb::tbb_hash_compare<Jungle::Tiger>::hash(Jungle::Tiger()); // Instantiation chain finds tbb_hasher via Argument Dependent Lookup
}
#endif

struct UserDefinedKeyType {
};

namespace tbb {
    // Test whether tbb_hash_compare can be partially specialized as stated in Reference manual.
    template<> struct tbb_hash_compare<UserDefinedKeyType> {
        size_t hash( UserDefinedKeyType ) const {return 0;}
        bool equal( UserDefinedKeyType /*x*/, UserDefinedKeyType /*y*/ ) {return true;}
    };
}

#include "harness_runtime_loader.h"

tbb::concurrent_hash_map<UserDefinedKeyType,int> TestInstantiationWithUserDefinedKeyType;

// Test whether a sufficient set of headers were included to instantiate a concurrent_hash_map. OSS Bug #120 (& #130):
// http://www.threadingbuildingblocks.org/bug_desc.php?id=120
tbb::concurrent_hash_map<std::pair<std::pair<int,std::string>,const char*>,int> TestInstantiation;

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/atomic.h"
#include "tbb/tick_count.h"
#include "harness.h"
#include "harness_allocator.h"

class MyException : public std::bad_alloc {
public:
    virtual const char *what() const throw() __TBB_override { return "out of items limit"; }
    virtual ~MyException() throw() {}
};

/** Has tightly controlled interface so that we can verify
    that concurrent_hash_map uses only the required interface. */
class MyKey {
private:
    void operator=( const MyKey&  );    // Deny access
    int key;
    friend class MyHashCompare;
    friend class YourHashCompare;
public:
    static MyKey make( int i ) {
        MyKey result;
        result.key = i;
        return result;
    }
    int value_of() const {return key;}
};
//TODO: unify with Harness::Foo ?
tbb::atomic<long> MyDataCount;
long MyDataCountLimit = 0;

class MyData {
protected:
    friend class MyData2;
    int data;
    enum state_t {
        LIVE=0x1234,
        DEAD=0x5678
    } my_state;
    void operator=( const MyData& );    // Deny access
public:
    MyData(int i = 0) {
        my_state = LIVE;
        data = i;
        if(MyDataCountLimit && MyDataCount + 1 >= MyDataCountLimit)
            __TBB_THROW( MyException() );
        ++MyDataCount;
    }
    MyData( const MyData& other ) {
        ASSERT( other.my_state==LIVE, NULL );
        my_state = LIVE;
        data = other.data;
        if(MyDataCountLimit && MyDataCount + 1 >= MyDataCountLimit)
            __TBB_THROW( MyException() );
        ++MyDataCount;
    }
    ~MyData() {
        --MyDataCount;
        my_state = DEAD;
    }
    static MyData make( int i ) {
        MyData result;
        result.data = i;
        return result;
    }
    int value_of() const {
        ASSERT( my_state==LIVE, NULL );
        return data;
    }
    void set_value( int i ) {
        ASSERT( my_state==LIVE, NULL );
        data = i;
    }
    bool operator==( const MyData& other ) const {
        ASSERT( other.my_state==LIVE, NULL );
        ASSERT( my_state==LIVE, NULL );
        return data == other.data;
    }
};

class MyData2 : public MyData {
public:
    MyData2( ) {}
    MyData2( const MyData& other ) {
        ASSERT( other.my_state==LIVE, NULL );
        ASSERT( my_state==LIVE, NULL );
        data = other.data;
    }
    void operator=( const MyData& other ) {
        ASSERT( other.my_state==LIVE, NULL );
        ASSERT( my_state==LIVE, NULL );
        data = other.data;
    }
    void operator=( const MyData2& other ) {
        ASSERT( other.my_state==LIVE, NULL );
        ASSERT( my_state==LIVE, NULL );
        data = other.data;
    }
    bool operator==( const MyData2& other ) const {
        ASSERT( other.my_state==LIVE, NULL );
        ASSERT( my_state==LIVE, NULL );
        return data == other.data;
    }
};

class MyHashCompare {
public:
    bool equal( const MyKey& j, const MyKey& k ) const {
        return j.key==k.key;
    }
    unsigned long hash( const MyKey& k ) const {
        return k.key;
    }
};

class YourHashCompare {
public:
    bool equal( const MyKey& j, const MyKey& k ) const {
        return j.key==k.key;
    }
    unsigned long hash( const MyKey& ) const {
        return 1;
    }
};

typedef local_counting_allocator<std::allocator<MyData> > MyAllocator;
typedef tbb::concurrent_hash_map<MyKey,MyData,MyHashCompare,MyAllocator> MyTable;
typedef tbb::concurrent_hash_map<MyKey,MyData2,MyHashCompare> MyTable2;
typedef tbb::concurrent_hash_map<MyKey,MyData,YourHashCompare> YourTable;

template<typename MyTable>
inline void CheckAllocator(MyTable &table, size_t expected_allocs, size_t expected_frees, bool exact = true) {
    size_t items_allocated = table.get_allocator().items_allocated, items_freed = table.get_allocator().items_freed;
    size_t allocations = table.get_allocator().allocations, frees = table.get_allocator().frees;
    REMARK("checking allocators: items %u/%u, allocs %u/%u\n",
            unsigned(items_allocated), unsigned(items_freed), unsigned(allocations), unsigned(frees) );
    ASSERT( items_allocated == allocations, NULL); ASSERT( items_freed == frees, NULL);
    if(exact) {
        ASSERT( allocations == expected_allocs, NULL); ASSERT( frees == expected_frees, NULL);
    } else {
        ASSERT( allocations >= expected_allocs, NULL); ASSERT( frees >= expected_frees, NULL);
        ASSERT( allocations - frees == expected_allocs - expected_frees, NULL );
    }
}

inline bool UseKey( size_t i ) {
    return (i&3)!=3;
}

struct Insert {
    static void apply( MyTable& table, int i ) {
        if( UseKey(i) ) {
            if( i&4 ) {
                MyTable::accessor a;
                table.insert( a, MyKey::make(i) );
                if( i&1 )
                    (*a).second.set_value(i*i);
                else
                    a->second.set_value(i*i);
            } else
                if( i&1 ) {
                    MyTable::accessor a;
                    table.insert( a, std::make_pair(MyKey::make(i), MyData(i*i)) );
                    ASSERT( (*a).second.value_of()==i*i, NULL );
                } else {
                    MyTable::const_accessor ca;
                    table.insert( ca, std::make_pair(MyKey::make(i), MyData(i*i)) );
                    ASSERT( ca->second.value_of()==i*i, NULL );
                }
        }
    }
};

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include "test_container_move_support.h"
typedef tbb::concurrent_hash_map<MyKey,Foo,MyHashCompare> DataStateTrackedTable;

struct RvalueInsert {
    static void apply( DataStateTrackedTable& table, int i ) {
        DataStateTrackedTable::accessor a;
        ASSERT( (table.insert( a, std::make_pair(MyKey::make(i), Foo(i + 1)))),"already present while should not ?" );
        ASSERT( (*a).second == i + 1, NULL );
        ASSERT( (*a).second.state == Harness::StateTrackableBase::MoveInitialized, "");
    }
};

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
struct Emplace {
    static void apply( DataStateTrackedTable& table, int i ) {
        DataStateTrackedTable::accessor a;
        ASSERT( (table.emplace( a, MyKey::make(i), (i + 1))),"already present while should not ?" );
        ASSERT( (*a).second == i + 1, NULL );
        ASSERT( (*a).second.state == Harness::StateTrackableBase::DirectInitialized, "");
    }
};
#endif // __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

#if __TBB_INITIALIZER_LISTS_PRESENT
struct InsertInitList {
    static void apply( MyTable& table, int i ) {
        if ( UseKey( i ) ) {
            // TODO: investigate why the following sequence causes an additional allocation sometimes:
            // table.insert( MyTable::value_type( MyKey::make( i ), i*i ) );
            // table.insert( MyTable::value_type( MyKey::make( i ), i*i+1 ) );
            std::initializer_list<MyTable::value_type> il = { MyTable::value_type( MyKey::make( i ), i*i )/*, MyTable::value_type( MyKey::make( i ), i*i+1 ) */ };
            table.insert( il );
        }
    }
};
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */

struct Find {
    static void apply( MyTable& table, int i ) {
        MyTable::accessor a;
        const MyTable::accessor& ca = a;
        bool b = table.find( a, MyKey::make(i) );
        ASSERT( b==!a.empty(), NULL );
        if( b ) {
            if( !UseKey(i) )
                REPORT("Line %d: unexpected key %d present\n",__LINE__,i);
            AssertSameType( &*a, static_cast<MyTable::value_type*>(0) );
            ASSERT( ca->second.value_of()==i*i, NULL );
            ASSERT( (*ca).second.value_of()==i*i, NULL );
            if( i&1 )
                ca->second.set_value( ~ca->second.value_of() );
            else
                (*ca).second.set_value( ~ca->second.value_of() );
        } else {
            if( UseKey(i) )
                REPORT("Line %d: key %d missing\n",__LINE__,i);
        }
    }
};

struct FindConst {
    static void apply( const MyTable& table, int i ) {
        MyTable::const_accessor a;
        const MyTable::const_accessor& ca = a;
        bool b = table.find( a, MyKey::make(i) );
        ASSERT( b==(table.count(MyKey::make(i))>0), NULL );
        ASSERT( b==!a.empty(), NULL );
        ASSERT( b==UseKey(i), NULL );
        if( b ) {
            AssertSameType( &*ca, static_cast<const MyTable::value_type*>(0) );
            ASSERT( ca->second.value_of()==~(i*i), NULL );
            ASSERT( (*ca).second.value_of()==~(i*i), NULL );
        }
    }
};

tbb::atomic<int> EraseCount;

struct Erase {
    static void apply( MyTable& table, int i ) {
        bool b;
        if(i&4) {
            if(i&8) {
                MyTable::const_accessor a;
                b = table.find( a, MyKey::make(i) ) && table.erase( a );
            } else {
                MyTable::accessor a;
                b = table.find( a, MyKey::make(i) ) && table.erase( a );
            }
        } else
            b = table.erase( MyKey::make(i) );
        if( b ) ++EraseCount;
        ASSERT( table.count(MyKey::make(i)) == 0, NULL );
    }
};

static const int IE_SIZE = 2;
tbb::atomic<YourTable::size_type> InsertEraseCount[IE_SIZE];

struct InsertErase  {
    static void apply( YourTable& table, int i ) {
        if ( i%3 ) {
            int key = i%IE_SIZE;
            if ( table.insert( std::make_pair(MyKey::make(key), MyData2()) ) )
                ++InsertEraseCount[key];
        } else {
            int key = i%IE_SIZE;
            if( i&1 ) {
                YourTable::accessor res;
                if(table.find( res, MyKey::make(key) ) && table.erase( res ) )
                    --InsertEraseCount[key];
            } else {
                YourTable::const_accessor res;
                if(table.find( res, MyKey::make(key) ) && table.erase( res ) )
                    --InsertEraseCount[key];
            }
        }
    }
};

// Test for the deadlock discussed at:
// http://softwarecommunity.intel.com/isn/Community/en-US/forums/permalink/30253302/30253302/ShowThread.aspx#30253302
struct InnerInsert {
    static void apply( YourTable& table, int i ) {
        YourTable::accessor a1, a2;
        if(i&1) __TBB_Yield();
        table.insert( a1, MyKey::make(1) );
        __TBB_Yield();
        table.insert( a2, MyKey::make(1 + (1<<30)) ); // the same chain
        table.erase( a2 ); // if erase by key it would lead to deadlock for single thread
    }
};

#include "harness_barrier.h"
// Test for the misuse of constness
struct FakeExclusive : NoAssign {
    Harness::SpinBarrier& barrier;
    YourTable& table;
    FakeExclusive(Harness::SpinBarrier& b, YourTable&t) : barrier(b), table(t) {}
    void operator()( int i ) const {
        if(i) {
            YourTable::const_accessor real_ca;
            // const accessor on non-const table acquired as reader (shared)
            ASSERT( table.find(real_ca,MyKey::make(1)), NULL );
            barrier.wait(); // item can be erased
            Harness::Sleep(10); // let it enter the erase
            real_ca->second.value_of(); // check the state while holding accessor
        } else {
            YourTable::accessor fake_ca;
            const YourTable &const_table = table;
            // non-const accessor on const table acquired as reader (shared)
            ASSERT( const_table.find(fake_ca,MyKey::make(1)), NULL );
            barrier.wait(); // readers acquired
            // can mistakenly remove the item while other readers still refers to it
            table.erase( fake_ca );
        }
    }
};

template<typename Op, typename MyTable>
class TableOperation: NoAssign {
    MyTable& my_table;
public:
    void operator()( const tbb::blocked_range<int>& range ) const {
        for( int i=range.begin(); i!=range.end(); ++i )
            Op::apply(my_table,i);
    }
    TableOperation( MyTable& table ) : my_table(table) {}
};

template<typename Op, typename TableType>
void DoConcurrentOperations( TableType& table, int n, const char* what, int nthread ) {
    REMARK("testing %s with %d threads\n",what,nthread);
    tbb::tick_count t0 = tbb::tick_count::now();
    tbb::parallel_for( tbb::blocked_range<int>(0,n,100), TableOperation<Op,TableType>(table) );
    tbb::tick_count t1 = tbb::tick_count::now();
    REMARK("time for %s = %g with %d threads\n",what,(t1-t0).seconds(),nthread);
}

//! Test traversing the table with an iterator.
void TraverseTable( MyTable& table, size_t n, size_t expected_size ) {
    REMARK("testing traversal\n");
    size_t actual_size = table.size();
    ASSERT( actual_size==expected_size, NULL );
    size_t count = 0;
    bool* array = new bool[n];
    memset( array, 0, n*sizeof(bool) );
    const MyTable& const_table = table;
    MyTable::const_iterator ci = const_table.begin();
    for( MyTable::iterator i = table.begin(); i!=table.end(); ++i ) {
        // Check iterator
        int k = i->first.value_of();
        ASSERT( UseKey(k), NULL );
        ASSERT( (*i).first.value_of()==k, NULL );
        ASSERT( 0<=k && size_t(k)<n, "out of bounds key" );
        ASSERT( !array[k], "duplicate key" );
        array[k] = true;
        ++count;

        // Check lower/upper bounds
        std::pair<MyTable::iterator, MyTable::iterator> er = table.equal_range(i->first);
        std::pair<MyTable::const_iterator, MyTable::const_iterator> cer = const_table.equal_range(i->first);
        ASSERT(cer.first == er.first && cer.second == er.second, NULL);
        ASSERT(cer.first == i, NULL);
        ASSERT(std::distance(cer.first, cer.second) == 1, NULL);

        // Check const_iterator
        MyTable::const_iterator cic = ci++;
        ASSERT( cic->first.value_of()==k, NULL );
        ASSERT( (*cic).first.value_of()==k, NULL );
    }
    ASSERT( ci==const_table.end(), NULL );
    delete[] array;
    if( count!=expected_size ) {
        REPORT("Line %d: count=%ld but should be %ld\n",__LINE__,long(count),long(expected_size));
    }
}

typedef tbb::atomic<unsigned char> AtomicByte;

template<typename RangeType>
struct ParallelTraverseBody: NoAssign {
    const size_t n;
    AtomicByte* const array;
    ParallelTraverseBody( AtomicByte array_[], size_t n_ ) :
        n(n_),
        array(array_)
    {}
    void operator()( const RangeType& range ) const {
        for( typename RangeType::iterator i = range.begin(); i!=range.end(); ++i ) {
            int k = i->first.value_of();
            ASSERT( 0<=k && size_t(k)<n, NULL );
            ++array[k];
        }
    }
};

void Check( AtomicByte array[], size_t n, size_t expected_size ) {
    if( expected_size )
        for( size_t k=0; k<n; ++k ) {
            if( array[k] != int(UseKey(k)) ) {
                REPORT("array[%d]=%d != %d=UseKey(%d)\n",
                       int(k), int(array[k]), int(UseKey(k)), int(k));
                ASSERT(false,NULL);
            }
        }
}

//! Test travering the tabel with a parallel range
void ParallelTraverseTable( MyTable& table, size_t n, size_t expected_size ) {
    REMARK("testing parallel traversal\n");
    ASSERT( table.size()==expected_size, NULL );
    AtomicByte* array = new AtomicByte[n];

    memset( array, 0, n*sizeof(AtomicByte) );
    MyTable::range_type r = table.range(10);
    tbb::parallel_for( r, ParallelTraverseBody<MyTable::range_type>( array, n ));
    Check( array, n, expected_size );

    const MyTable& const_table = table;
    memset( array, 0, n*sizeof(AtomicByte) );
    MyTable::const_range_type cr = const_table.range(10);
    tbb::parallel_for( cr, ParallelTraverseBody<MyTable::const_range_type>( array, n ));
    Check( array, n, expected_size );

    delete[] array;
}

void TestInsertFindErase( int nthread ) {
    int n=250000;

    // compute m = number of unique keys
    int m = 0;
    for( int i=0; i<n; ++i )
        m += UseKey(i);

    MyAllocator a; a.items_freed = a.frees = 100;
    ASSERT( MyDataCount==0, NULL );
    MyTable table(a);
    TraverseTable(table,n,0);
    ParallelTraverseTable(table,n,0);
    CheckAllocator(table, 0, 100);

    int expected_allocs = 0, expected_frees = 100;
#if __TBB_INITIALIZER_LISTS_PRESENT
    for ( int i = 0; i < 2; ++i ) {
        if ( i==0 )
            DoConcurrentOperations<InsertInitList, MyTable>( table, n, "insert(std::initializer_list)", nthread );
        else
#endif
            DoConcurrentOperations<Insert, MyTable>( table, n, "insert", nthread );
        ASSERT( MyDataCount == m, NULL );
        TraverseTable( table, n, m );
        ParallelTraverseTable( table, n, m );
        expected_allocs += m;
        CheckAllocator( table, expected_allocs, expected_frees );

        DoConcurrentOperations<Find, MyTable>( table, n, "find", nthread );
        ASSERT( MyDataCount == m, NULL );
        CheckAllocator( table, expected_allocs, expected_frees );

        DoConcurrentOperations<FindConst, MyTable>( table, n, "find(const)", nthread );
        ASSERT( MyDataCount == m, NULL );
        CheckAllocator( table, expected_allocs, expected_frees );

        EraseCount = 0;
        DoConcurrentOperations<Erase, MyTable>( table, n, "erase", nthread );
        ASSERT( EraseCount == m, NULL );
        ASSERT( MyDataCount == 0, NULL );
        TraverseTable( table, n, 0 );
        expected_frees += m;
        CheckAllocator( table, expected_allocs, expected_frees );

        bad_hashing = true;
        table.clear();
        bad_hashing = false;
#if __TBB_INITIALIZER_LISTS_PRESENT
    }
#endif

    if(nthread > 1) {
        YourTable ie_table;
        for( int i=0; i<IE_SIZE; ++i )
            InsertEraseCount[i] = 0;
        DoConcurrentOperations<InsertErase,YourTable>(ie_table,n/2,"insert_erase",nthread);
        for( int i=0; i<IE_SIZE; ++i )
            ASSERT( InsertEraseCount[i]==ie_table.count(MyKey::make(i)), NULL );

        DoConcurrentOperations<InnerInsert,YourTable>(ie_table,2000,"inner insert",nthread);
        Harness::SpinBarrier barrier(nthread);
        REMARK("testing erase on fake exclusive accessor\n");
        NativeParallelFor( nthread, FakeExclusive(barrier, ie_table));
    }
}

volatile int Counter;

class AddToTable: NoAssign {
    MyTable& my_table;
    const int my_nthread;
    const int my_m;
public:
    AddToTable( MyTable& table, int nthread, int m ) : my_table(table), my_nthread(nthread), my_m(m) {}
    void operator()( int ) const {
        for( int i=0; i<my_m; ++i ) {
            // Busy wait to synchronize threads
            int j = 0;
            while( Counter<i ) {
                if( ++j==1000000 ) {
                    // If Counter<i after a million iterations, then we almost surely have
                    // more logical threads than physical threads, and should yield in
                    // order to let suspended logical threads make progress.
                    j = 0;
                    __TBB_Yield();
                }
            }
            // Now all threads attempt to simultaneously insert a key.
            int k;
            {
                MyTable::accessor a;
                MyKey key = MyKey::make(i);
                if( my_table.insert( a, key ) )
                    a->second.set_value( 1 );
                else
                    a->second.set_value( a->second.value_of()+1 );
                k = a->second.value_of();
            }
            if( k==my_nthread )
                Counter=i+1;
        }
    }
};

class RemoveFromTable: NoAssign {
    MyTable& my_table;
    const int my_nthread;
    const int my_m;
public:
    RemoveFromTable( MyTable& table, int nthread, int m ) : my_table(table), my_nthread(nthread), my_m(m) {}
    void operator()(int) const {
        for( int i=0; i<my_m; ++i ) {
            bool b;
            if(i&4) {
                if(i&8) {
                    MyTable::const_accessor a;
                    b = my_table.find( a, MyKey::make(i) ) && my_table.erase( a );
                } else {
                    MyTable::accessor a;
                    b = my_table.find( a, MyKey::make(i) ) && my_table.erase( a );
                }
            } else
                b = my_table.erase( MyKey::make(i) );
            if( b ) ++EraseCount;
        }
    }
};

//! Test for memory leak in concurrent_hash_map (TR #153).
void TestConcurrency( int nthread ) {
    REMARK("testing multiple insertions/deletions of same key with %d threads\n", nthread);
    {
        ASSERT( MyDataCount==0, NULL );
        MyTable table;
        const int m = 1000;
        Counter = 0;
        tbb::tick_count t0 = tbb::tick_count::now();
        NativeParallelFor( nthread, AddToTable(table,nthread,m) );
        tbb::tick_count t1 = tbb::tick_count::now();
        REMARK("time for %u insertions = %g with %d threads\n",unsigned(MyDataCount),(t1-t0).seconds(),nthread);
        ASSERT( MyDataCount==m, "memory leak detected" );

        EraseCount = 0;
        t0 = tbb::tick_count::now();
        NativeParallelFor( nthread, RemoveFromTable(table,nthread,m) );
        t1 = tbb::tick_count::now();
        REMARK("time for %u deletions = %g with %d threads\n",unsigned(EraseCount),(t1-t0).seconds(),nthread);
        ASSERT( MyDataCount==0, "memory leak detected" );
        ASSERT( EraseCount==m, "return value of erase() is broken" );

        CheckAllocator(table, m, m, /*exact*/nthread <= 1);
    }
    ASSERT( MyDataCount==0, "memory leak detected" );
}

void TestTypes() {
    AssertSameType( static_cast<MyTable::key_type*>(0), static_cast<MyKey*>(0) );
    AssertSameType( static_cast<MyTable::mapped_type*>(0), static_cast<MyData*>(0) );
    AssertSameType( static_cast<MyTable::value_type*>(0), static_cast<std::pair<const MyKey,MyData>*>(0) );
    AssertSameType( static_cast<MyTable::accessor::value_type*>(0), static_cast<MyTable::value_type*>(0) );
    AssertSameType( static_cast<MyTable::const_accessor::value_type*>(0), static_cast<const MyTable::value_type*>(0) );
    AssertSameType( static_cast<MyTable::size_type*>(0), static_cast<size_t*>(0) );
    AssertSameType( static_cast<MyTable::difference_type*>(0), static_cast<ptrdiff_t*>(0) );
}

template<typename Iterator, typename T>
void TestIteratorTraits() {
    AssertSameType( static_cast<typename Iterator::difference_type*>(0), static_cast<ptrdiff_t*>(0) );
    AssertSameType( static_cast<typename Iterator::value_type*>(0), static_cast<T*>(0) );
    AssertSameType( static_cast<typename Iterator::pointer*>(0), static_cast<T**>(0) );
    AssertSameType( static_cast<typename Iterator::iterator_category*>(0), static_cast<std::forward_iterator_tag*>(0) );
    T x;
    typename Iterator::reference xr = x;
    typename Iterator::pointer xp = &x;
    ASSERT( &xr==xp, NULL );
}

template<typename Iterator1, typename Iterator2>
void TestIteratorAssignment( Iterator2 j ) {
    Iterator1 i(j), k;
    ASSERT( i==j, NULL ); ASSERT( !(i!=j), NULL );
    k = j;
    ASSERT( k==j, NULL ); ASSERT( !(k!=j), NULL );
}

template<typename Range1, typename Range2>
void TestRangeAssignment( Range2 r2 ) {
    Range1 r1(r2); r1 = r2;
}
//------------------------------------------------------------------------
// Test for copy constructor and assignment
//------------------------------------------------------------------------

template<typename MyTable>
static void FillTable( MyTable& x, int n ) {
    for( int i=1; i<=n; ++i ) {
        MyKey key( MyKey::make(-i) ); // hash values must not be specified in direct order
        typename MyTable::accessor a;
        bool b = x.insert(a,key);
        ASSERT(b, NULL);
        a->second.set_value( i*i );
    }
}

template<typename MyTable>
static void CheckTable( const MyTable& x, int n ) {
    ASSERT( x.size()==size_t(n), "table is different size than expected" );
    ASSERT( x.empty()==(n==0), NULL );
    ASSERT( x.size()<=x.max_size(), NULL );
    for( int i=1; i<=n; ++i ) {
        MyKey key( MyKey::make(-i) );
        typename MyTable::const_accessor a;
        bool b = x.find(a,key);
        ASSERT( b, NULL );
        ASSERT( a->second.value_of()==i*i, NULL );
    }
    int count = 0;
    int key_sum = 0;
    for( typename MyTable::const_iterator i(x.begin()); i!=x.end(); ++i ) {
        ++count;
        key_sum += -i->first.value_of();
    }
    ASSERT( count==n, NULL );
    ASSERT( key_sum==n*(n+1)/2, NULL );
}

static void TestCopy() {
    REMARK("testing copy\n");
    MyTable t1;
    for( int i=0; i<10000; i=(i<100 ? i+1 : i*3) ) {
        MyDataCount = 0;

        FillTable(t1,i);
        // Do not call CheckTable(t1,i) before copying, it enforces rehashing

        MyTable t2(t1);
        // Check that copy constructor did not mangle source table.
        CheckTable(t1,i);
        swap(t1, t2);
        CheckTable(t1,i);
        ASSERT( !(t1 != t2), NULL );

        // Clear original table
        t2.clear();
        swap(t2, t1);
        CheckTable(t1,0);

        // Verify that copy of t1 is correct, even after t1 is cleared.
        CheckTable(t2,i);
        t2.clear();
        t1.swap( t2 );
        CheckTable(t1,0);
        CheckTable(t2,0);
        ASSERT( MyDataCount==0, "data leak?" );
    }
}

void TestAssignment() {
    REMARK("testing assignment\n");
    for( int i=0; i<1000; i=(i<30 ? i+1 : i*5) ) {
        for( int j=0; j<1000; j=(j<30 ? j+1 : j*7) ) {
            MyTable t1;
            MyTable t2;
            FillTable(t1,i);
            FillTable(t2,j);
            ASSERT( (t1 == t2) == (i == j), NULL );
            CheckTable(t2,j);

            MyTable& tref = t2=t1;
            ASSERT( &tref==&t2, NULL );
            ASSERT( t1 == t2, NULL );
            CheckTable(t1,i);
            CheckTable(t2,i);

            t1.clear();
            CheckTable(t1,0);
            CheckTable(t2,i);
            ASSERT( MyDataCount==i, "data leak?" );

            t2.clear();
            CheckTable(t1,0);
            CheckTable(t2,0);
            ASSERT( MyDataCount==0, "data leak?" );
        }
    }
}

void TestIteratorsAndRanges() {
    REMARK("testing iterators compliance\n");
    TestIteratorTraits<MyTable::iterator,MyTable::value_type>();
    TestIteratorTraits<MyTable::const_iterator,const MyTable::value_type>();

    MyTable v;
    MyTable const &u = v;

    TestIteratorAssignment<MyTable::const_iterator>( u.begin() );
    TestIteratorAssignment<MyTable::const_iterator>( v.begin() );
    TestIteratorAssignment<MyTable::iterator>( v.begin() );
    // doesn't compile as expected: TestIteratorAssignment<typename V::iterator>( u.begin() );

    // check for non-existing
    ASSERT(v.equal_range(MyKey::make(-1)) == std::make_pair(v.end(), v.end()), NULL);
    ASSERT(u.equal_range(MyKey::make(-1)) == std::make_pair(u.end(), u.end()), NULL);

    REMARK("testing ranges compliance\n");
    TestRangeAssignment<MyTable::const_range_type>( u.range() );
    TestRangeAssignment<MyTable::const_range_type>( v.range() );
    TestRangeAssignment<MyTable::range_type>( v.range() );
    // doesn't compile as expected: TestRangeAssignment<typename V::range_type>( u.range() );

    REMARK("testing construction and insertion from iterators range\n");
    FillTable( v, 1000 );
    MyTable2 t(v.begin(), v.end());
    v.rehash();
    CheckTable(t, 1000);
    t.insert(v.begin(), v.end()); // do nothing
    CheckTable(t, 1000);
    t.clear();
    t.insert(v.begin(), v.end()); // restore
    CheckTable(t, 1000);

    REMARK("testing comparison\n");
    typedef tbb::concurrent_hash_map<MyKey,MyData2,YourHashCompare,MyAllocator> YourTable1;
    typedef tbb::concurrent_hash_map<MyKey,MyData2,YourHashCompare> YourTable2;
    YourTable1 t1;
    FillTable( t1, 10 );
    CheckTable(t1, 10 );
    YourTable2 t2(t1.begin(), t1.end());
    MyKey key( MyKey::make(-5) ); MyData2 data;
    ASSERT(t2.erase(key), NULL);
    YourTable2::accessor a;
    ASSERT(t2.insert(a, key), NULL);
    data.set_value(0);   a->second = data;
    ASSERT( t1 != t2, NULL);
    data.set_value(5*5); a->second = data;
    ASSERT( t1 == t2, NULL);
}

void TestRehash() {
    REMARK("testing rehashing\n");
    MyTable w;
    w.insert( std::make_pair(MyKey::make(-5), MyData()) );
    w.rehash(); // without this, assertion will fail
    MyTable::iterator it = w.begin();
    int i = 0; // check for non-rehashed buckets
    for( ; it != w.end(); i++ )
        w.count( (it++)->first );
    ASSERT( i == 1, NULL );
    for( i=0; i<1000; i=(i<29 ? i+1 : i*2) ) {
        for( int j=max(256+i, i*2); j<10000; j*=3 ) {
            MyTable v;
            FillTable( v, i );
            ASSERT(int(v.size()) == i, NULL);
            ASSERT(int(v.bucket_count()) <= j, NULL);
            v.rehash( j );
            ASSERT(int(v.bucket_count()) >= j, NULL);
            CheckTable( v, i );
        }
    }
}

#if TBB_USE_EXCEPTIONS
void TestExceptions() {
    typedef local_counting_allocator<tbb::tbb_allocator<MyData2> > allocator_t;
    typedef tbb::concurrent_hash_map<MyKey,MyData2,MyHashCompare,allocator_t> ThrowingTable;
    enum methods {
        zero_method = 0,
        ctor_copy, op_assign, op_insert,
        all_methods
    };
    REMARK("testing exception-safety guarantees\n");
    ThrowingTable src;
    FillTable( src, 1000 );
    ASSERT( MyDataCount==1000, NULL );

    try {
        for(int t = 0; t < 2; t++) // exception type
        for(int m = zero_method+1; m < all_methods; m++)
        {
            allocator_t a;
            if(t) MyDataCountLimit = 101;
            else a.set_limits(101);
            ThrowingTable victim(a);
            MyDataCount = 0;

            try {
                switch(m) {
                case ctor_copy: {
                        ThrowingTable acopy(src, a);
                    } break;
                case op_assign: {
                        victim = src;
                    } break;
                case op_insert: {
                        FillTable( victim, 1000 );
                    } break;
                default:;
                }
                ASSERT(false, "should throw an exception");
            } catch(std::bad_alloc &e) {
                MyDataCountLimit = 0;
                size_t size = victim.size();
                switch(m) {
                case op_assign:
                    ASSERT( MyDataCount==100, "data leak?" );
                    ASSERT( size>=100, NULL );
                    CheckAllocator(victim, 100+t, t);
                case ctor_copy:
                    CheckTable(src, 1000);
                    break;
                case op_insert:
                    ASSERT( size==size_t(100-t), NULL );
                    ASSERT( MyDataCount==100-t, "data leak?" );
                    CheckTable(victim, 100-t);
                    CheckAllocator(victim, 100, t);
                    break;

                default:; // nothing to check here
                }
                REMARK("Exception %d: %s\t- ok ()\n", m, e.what());
            }
            catch ( ... ) {
                ASSERT ( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unrecognized exception" );
            }
        }
    } catch(...) {
        ASSERT(false, "unexpected exception");
    }
    src.clear(); MyDataCount = 0;
}
#endif /* TBB_USE_EXCEPTIONS */


#if __TBB_INITIALIZER_LISTS_PRESENT
#include "test_initializer_list.h"

struct test_insert {
    template<typename container_type, typename element_type>
    static void do_test( std::initializer_list<element_type> il, container_type const& expected ) {
        container_type vd;
        vd.insert( il );
        ASSERT( vd == expected, "inserting with an initializer list failed" );
    }
};

void TestInitList(){
    using namespace initializer_list_support_tests;
    REMARK("testing initializer_list methods \n");

    typedef tbb::concurrent_hash_map<int,int> ch_map_type;
    std::initializer_list<ch_map_type::value_type> pairs_il = {{1,1},{2,2},{3,3},{4,4},{5,5}};

    TestInitListSupportWithoutAssign<ch_map_type, test_insert>( pairs_il );
    TestInitListSupportWithoutAssign<ch_map_type, test_insert>( {} );
}
#endif //if __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_RANGE_BASED_FOR_PRESENT
#include "test_range_based_for.h"

void TestRangeBasedFor(){
    using namespace range_based_for_support_tests;

    REMARK("testing range based for loop compatibility \n");
    typedef tbb::concurrent_hash_map<int,int> ch_map;
    ch_map a_ch_map;

    const int sequence_length = 100;
    for (int i = 1; i <= sequence_length; ++i){
        a_ch_map.insert(ch_map::value_type(i,i));
    }

    ASSERT( range_based_for_accumulate(a_ch_map, pair_second_summer(), 0) == gauss_summ_of_int_sequence(sequence_length), "incorrect accumulated value generated via range based for ?");
}
#endif //if __TBB_RANGE_BASED_FOR_PRESENT

#include "harness_defs.h"

// The helper to run a test only when a default construction is present.
template <bool default_construction_present> struct do_default_construction_test {
    template<typename FuncType> void operator() ( FuncType func ) const { func(); }
};
template <> struct do_default_construction_test<false> {
    template<typename FuncType> void operator()( FuncType ) const {}
};

template <typename Table>
class test_insert_by_key : NoAssign {
    typedef typename Table::value_type value_type;
    Table &my_c;
    const value_type &my_value;
public:
    test_insert_by_key( Table &c, const value_type &value ) : my_c(c), my_value(value) {}
    void operator()() const {
        {
            typename Table::accessor a;
            ASSERT( my_c.insert( a, my_value.first ), NULL );
            ASSERT( Harness::IsEqual()(a->first, my_value.first), NULL );
            a->second = my_value.second;
        } {
            typename Table::const_accessor ca;
            ASSERT( !my_c.insert( ca, my_value.first ), NULL );
            ASSERT( Harness::IsEqual()(ca->first, my_value.first), NULL);
            ASSERT( Harness::IsEqual()(ca->second, my_value.second), NULL);
        }
    }
};

#include <vector>
#include <list>
#include <algorithm>
#if __TBB_CPP11_REFERENCE_WRAPPER_PRESENT
#include <functional>
#endif

template <typename Table, typename Iterator, typename Range = typename Table::range_type>
class test_range : NoAssign {
    typedef typename Table::value_type value_type;
    Table &my_c;
    const std::list<value_type> &my_lst;
    std::vector< tbb::atomic<bool> >& my_marks;
public:
    test_range( Table &c, const std::list<value_type> &lst, std::vector< tbb::atomic<bool> > &marks ) : my_c(c), my_lst(lst), my_marks(marks) {
        std::fill( my_marks.begin(), my_marks.end(), false );
    }
    void operator()( const Range &r ) const { do_test_range( r.begin(), r.end() ); }
    void do_test_range( Iterator i, Iterator j ) const {
        for ( Iterator it = i; it != j; ) {
            Iterator it_prev = it++;
            typename std::list<value_type>::const_iterator it2 = std::search( my_lst.begin(), my_lst.end(), it_prev, it, Harness::IsEqual() );
            ASSERT( it2 != my_lst.end(), NULL );
            typename std::list<value_type>::difference_type dist = std::distance( my_lst.begin(), it2 );
            ASSERT( !my_marks[dist], NULL );
            my_marks[dist] = true;
        }
    }
};

template <bool default_construction_present, typename Table>
class check_value : NoAssign {
    typedef typename Table::const_iterator const_iterator;
    typedef typename Table::iterator iterator;
    typedef typename Table::size_type size_type;
    Table &my_c;
public:
    check_value( Table &c ) : my_c(c) {}
    void operator()(const typename Table::value_type &value ) {
        const Table &const_c = my_c;
        ASSERT( my_c.count( value.first ) == 1, NULL );
        { // tests with a const accessor.
            typename Table::const_accessor ca;
            // find
            ASSERT( my_c.find( ca, value.first ), NULL);
            ASSERT( !ca.empty() , NULL);
            ASSERT( Harness::IsEqual()(ca->first, value.first), NULL );
            ASSERT( Harness::IsEqual()(ca->second, value.second), NULL );
            // erase
            ASSERT( my_c.erase( ca ), NULL );
            ASSERT( my_c.count( value.first ) == 0, NULL );
            // insert (pair)
            ASSERT( my_c.insert( ca, value ), NULL);
            ASSERT( Harness::IsEqual()(ca->first, value.first), NULL );
            ASSERT( Harness::IsEqual()(ca->second, value.second), NULL );
        } { // tests with a non-const accessor.
            typename Table::accessor a;
            // find
            ASSERT( my_c.find( a, value.first ), NULL);
            ASSERT( !a.empty() , NULL);
            ASSERT( Harness::IsEqual()(a->first, value.first), NULL );
            ASSERT( Harness::IsEqual()(a->second, value.second), NULL );
            // erase
            ASSERT( my_c.erase( a ), NULL );
            ASSERT( my_c.count( value.first ) == 0, NULL );
            // insert
            ASSERT( my_c.insert( a, value ), NULL);
            ASSERT( Harness::IsEqual()(a->first, value.first), NULL );
            ASSERT( Harness::IsEqual()(a->second, value.second), NULL );
        }
        // erase by key
        ASSERT( my_c.erase( value.first ), NULL );
        ASSERT( my_c.count( value.first ) == 0, NULL );
        do_default_construction_test<default_construction_present>()(test_insert_by_key<Table>( my_c, value ));
        // insert by value
        ASSERT( my_c.insert( value ) != default_construction_present, NULL );
        // equal_range
        std::pair<iterator,iterator> r1 = my_c.equal_range( value.first );
        iterator r1_first_prev = r1.first++;
        ASSERT( Harness::IsEqual()( *r1_first_prev, value ) && Harness::IsEqual()( r1.first, r1.second ), NULL );
        std::pair<const_iterator,const_iterator> r2 = const_c.equal_range( value.first );
        const_iterator r2_first_prev = r2.first++;
        ASSERT( Harness::IsEqual()( *r2_first_prev, value ) && Harness::IsEqual()( r2.first, r2.second ), NULL );
    }
};

#include "tbb/task_scheduler_init.h"

template <typename Value, typename U = Value>
struct CompareTables {
    template <typename T>
    static bool IsEqual( const T& t1, const T& t2 ) {
        return (t1 == t2) && !(t1 != t2);
    }
};

#if __TBB_CPP11_SMART_POINTERS_PRESENT
template <typename U>
struct CompareTables< std::pair<const std::weak_ptr<U>, std::weak_ptr<U> > > {
    template <typename T>
    static bool IsEqual( const T&, const T& ) {
        /* do nothing for std::weak_ptr */
        return true;
    }
};
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

template <bool default_construction_present, typename Table>
void Examine( Table c, const std::list<typename Table::value_type> &lst) {
    typedef const Table const_table;
    typedef typename Table::const_iterator const_iterator;
    typedef typename Table::iterator iterator;
    typedef typename Table::value_type value_type;
    typedef typename Table::size_type size_type;

    ASSERT( !c.empty(), NULL );
    ASSERT( c.size() == lst.size(), NULL );
    ASSERT( c.max_size() >= c.size(), NULL );

    const check_value<default_construction_present,Table> cv(c);
    std::for_each( lst.begin(), lst.end(), cv );

    std::vector< tbb::atomic<bool> > marks( lst.size() );

    test_range<Table,iterator>( c, lst, marks ).do_test_range( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    test_range<const_table,const_iterator>( c, lst, marks ).do_test_range( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    tbb::task_scheduler_init init;

    typedef typename Table::range_type range_type;
    tbb::parallel_for( c.range(), test_range<Table,typename range_type::iterator,range_type>( c, lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    const_table const_c = c;
    ASSERT( CompareTables<value_type>::IsEqual( c, const_c ), NULL );

    typedef typename const_table::const_range_type const_range_type;
    tbb::parallel_for( c.range(), test_range<const_table,typename const_range_type::iterator,const_range_type>( const_c, lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    const size_type new_bucket_count = 2*c.bucket_count();
    c.rehash( new_bucket_count );
    ASSERT( c.bucket_count() >= new_bucket_count, NULL );

    Table c2;
    typename std::list<value_type>::const_iterator begin5 = lst.begin();
    std::advance( begin5, 5 );
    c2.insert( lst.begin(), begin5 );
    std::for_each( lst.begin(), begin5, check_value<default_construction_present, Table>( c2 ) );

    c2.swap( c );
    ASSERT( CompareTables<value_type>::IsEqual( c2, const_c ), NULL );
    ASSERT( c.size() == 5, NULL );
    std::for_each( lst.begin(), lst.end(), check_value<default_construction_present,Table>(c2) );

    tbb::swap( c, c2 );
    ASSERT( CompareTables<value_type>::IsEqual( c, const_c ), NULL );
    ASSERT( c2.size() == 5, NULL );

    c2.clear();
    ASSERT( CompareTables<value_type>::IsEqual( c2, Table() ), NULL );

    typename Table::allocator_type a = c.get_allocator();
    value_type *ptr = a.allocate(1);
    ASSERT( ptr, NULL );
    a.deallocate( ptr, 1 );
}

template <bool default_construction_present, typename Value>
void TypeTester( const std::list<Value> &lst ) {
    __TBB_ASSERT( lst.size() >= 5, "Array should have at least 5 elements" );
    typedef typename Value::first_type first_type;
    typedef typename Value::second_type second_type;
    typedef tbb::concurrent_hash_map<first_type,second_type> ch_map;
    // Construct an empty hash map.
    ch_map c1;
    c1.insert( lst.begin(), lst.end() );
    Examine<default_construction_present>( c1, lst );
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    // Constructor from initializer_list.
    typename std::list<Value>::const_iterator it = lst.begin();
    ch_map c2( {*it++, *it++, *it++} );
    c2.insert( it, lst.end() );
    Examine<default_construction_present>( c2, lst );
#endif
    // Copying constructor.
    ch_map c3(c1);
    Examine<default_construction_present>( c3, lst );
    // Construct with non-default allocator
    typedef tbb::concurrent_hash_map< first_type,second_type,tbb::tbb_hash_compare<first_type>,debug_allocator<Value> > ch_map_debug_alloc;
    ch_map_debug_alloc c4;
    c4.insert( lst.begin(), lst.end() );
    Examine<default_construction_present>( c4, lst );
    // Copying constructor for vector with different allocator type.
    ch_map_debug_alloc c5(c4);
    Examine<default_construction_present>( c5, lst );
    // Construction empty table with n preallocated buckets.
    ch_map c6( lst.size() );
    c6.insert( lst.begin(), lst.end() );
    Examine<default_construction_present>( c6, lst );
    ch_map_debug_alloc c7( lst.size() );
    c7.insert( lst.begin(), lst.end() );
    Examine<default_construction_present>( c7, lst );
    // Construction with copying iteration range and given allocator instance.
    ch_map c8( c1.begin(), c1.end() );
    Examine<default_construction_present>( c8, lst );
    debug_allocator<Value> allocator;
    ch_map_debug_alloc c9( lst.begin(), lst.end(), allocator );
    Examine<default_construction_present>( c9, lst );
}

#if __TBB_CPP11_SMART_POINTERS_PRESENT
namespace tbb {
    template<> struct tbb_hash_compare< const std::shared_ptr<int> > {
        static size_t hash( const std::shared_ptr<int>& ptr ) { return static_cast<size_t>( *ptr ) * interface5::internal::hash_multiplier; }
        static bool equal( const  std::shared_ptr<int>& ptr1, const  std::shared_ptr<int>& ptr2 ) { return ptr1 == ptr2; }
    };
    template<> struct tbb_hash_compare< const std::weak_ptr<int> > {
        static size_t hash( const std::weak_ptr<int>& ptr ) { return static_cast<size_t>( *ptr.lock() ) * interface5::internal::hash_multiplier; }
        static bool equal( const std::weak_ptr<int>& ptr1, const  std::weak_ptr<int>& ptr2 ) { return ptr1.lock() == ptr2.lock(); }
    };
}
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

void TestCPP11Types() {
    const int NUMBER = 10;

    typedef std::pair<const int, int> int_int_t;
    std::list<int_int_t> arrIntInt;
    for ( int i=0; i<NUMBER; ++i ) arrIntInt.push_back( int_int_t(i, NUMBER-i) );
    TypeTester</*default_construction_present = */true>( arrIntInt );

#if __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN
    typedef std::pair<const std::reference_wrapper<const int>, int> ref_int_t;
    std::list<ref_int_t> arrRefInt;
    for ( std::list<int_int_t>::iterator it = arrIntInt.begin(); it != arrIntInt.end(); ++it )
        arrRefInt.push_back( ref_int_t( it->first, it->second ) );
    TypeTester</*default_construction_present = */true>( arrRefInt );

    typedef std::pair< const int, std::reference_wrapper<int> > int_ref_t;
    std::list<int_ref_t> arrIntRef;
    for ( std::list<int_int_t>::iterator it = arrIntInt.begin(); it != arrIntInt.end(); ++it )
        arrIntRef.push_back( int_ref_t( it->first, it->second ) );
    TypeTester</*default_construction_present = */false>( arrIntRef );
#else
    REPORT("Known issue: C++11 reference wrapper tests are skipped.\n");
#endif /* __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN*/

    typedef std::pair< const int, tbb::atomic<int> > int_tbb_t;
    std::list<int_tbb_t> arrIntTbb;
    for ( int i=0; i<NUMBER; ++i ) {
        tbb::atomic<int> b;
        b = NUMBER-i;
        arrIntTbb.push_back( int_tbb_t(i, b) );
    }
    TypeTester</*default_construction_present = */true>( arrIntTbb );

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    typedef std::pair< const std::shared_ptr<int>, std::shared_ptr<int> > shr_shr_t;
    std::list<shr_shr_t> arrShrShr;
    for ( int i=0; i<NUMBER; ++i ) {
        const int NUMBER_minus_i = NUMBER - i;
        arrShrShr.push_back( shr_shr_t( std::make_shared<int>(i), std::make_shared<int>(NUMBER_minus_i) ) );
    }
    TypeTester< /*default_construction_present = */true>( arrShrShr );

    typedef std::pair< const std::weak_ptr<int>, std::weak_ptr<int> > wk_wk_t;
    std::list< wk_wk_t > arrWkWk;
    std::copy( arrShrShr.begin(), arrShrShr.end(), std::back_inserter(arrWkWk) );
    TypeTester< /*default_construction_present = */true>( arrWkWk );
#else
    REPORT("Known issue: C++11 smart pointer tests are skipped.\n");
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}

#if __TBB_CPP11_RVALUE_REF_PRESENT

struct hash_map_move_traits : default_container_traits {
    enum{ expected_number_of_items_to_allocate_for_steal_move = 0 };

    template<typename T>
    struct hash_compare {
        bool equal( const T& lhs, const T& rhs ) const {
            return lhs==rhs;
        }
        size_t hash( const T& k ) const {
            return tbb::tbb_hasher(k);
        }
    };
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_hash_map<element_type, element_type, hash_compare<element_type>, allocator_type > type;
    };

    typedef FooPairIterator init_iterator_type;
    template<typename hash_map_type, typename iterator>
    static bool equal(hash_map_type const& c, iterator begin, iterator end){
        bool equal_sizes = ( static_cast<size_t>(std::distance(begin, end)) == c.size() );
        if (!equal_sizes)
            return false;

        for (iterator it = begin; it != end; ++it ){
            if (c.count( (*it).first) == 0){
                return false;
            }
        }
        return true;
    }
};

void TestMoveSupport(){
    TestMoveConstructor<hash_map_move_traits>();
    TestConstructorWithMoveIterators<hash_map_move_traits>();
    TestMoveAssignOperator<hash_map_move_traits>();
#if TBB_USE_EXCEPTIONS
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure<hash_map_move_traits>();
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor<hash_map_move_traits>();
#else
    REPORT("Known issue: exception safety tests for C++11 move semantics support are skipped.\n");
#endif //TBB_USE_EXCEPTIONS
}
#else
void TestMoveSupport(){
    REPORT("Known issue: tests for C++11 move semantics support are skipped.\n");
}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
//------------------------------------------------------------------------
// Test driver
//------------------------------------------------------------------------
int TestMain () {
    if( MinThread<0 ) {
        REPORT("ERROR: must use at least one thread\n");
        exit(1);
    }
    if( MaxThread<2 ) MaxThread=2;

    // Do serial tests
    TestTypes();
    TestCopy();
    TestRehash();
    TestAssignment();
    TestIteratorsAndRanges();
#if __TBB_INITIALIZER_LISTS_PRESENT
    TestInitList();
#endif //__TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor();
#endif //#if __TBB_RANGE_BASED_FOR_PRESENT

#if TBB_USE_EXCEPTIONS
    TestExceptions();
#endif /* TBB_USE_EXCEPTIONS */

    TestMoveSupport();
    {
#if __TBB_CPP11_RVALUE_REF_PRESENT
        tbb::task_scheduler_init init( 1 );
        int n=250000;
        {
            DataStateTrackedTable table;
            DoConcurrentOperations<RvalueInsert, DataStateTrackedTable>( table, n, "rvalue ref insert", 1 );
        }
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
        {
            DataStateTrackedTable table;
            DoConcurrentOperations<Emplace, DataStateTrackedTable>( table, n, "emplace", 1 );
        }
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT
    }

    // Do concurrency tests.
    for( int nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        tbb::task_scheduler_init init( nthread );
        TestInsertFindErase( nthread );
        TestConcurrency( nthread );
    }
    // check linking
    if(bad_hashing) { //should be false
        tbb::internal::runtime_warning("none\nERROR: it must not be executed");
    }

    TestCPP11Types();

    return Harness::Done;
}
