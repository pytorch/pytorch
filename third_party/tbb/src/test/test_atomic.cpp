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

#include "harness_defs.h"

#if __TBB_TEST_SKIP_PIC_MODE || (__TBB_TEST_SKIP_GCC_BUILTINS_MODE && __TBB_TEST_SKIP_ICC_BUILTINS_MODE)
#include "harness.h"
int TestMain() {
    REPORT("Known issue: %s\n",
           __TBB_TEST_SKIP_PIC_MODE? "PIC mode is not supported" : "Compiler builtins for atomic operations aren't available");
    return Harness::Skipped;
}
#else

// Put tbb/atomic.h first, so if it is missing a prerequisite header, we find out about it.
// The tests here do *not* test for atomicity, just serial correctness. */

#include "tbb/atomic.h"
#include "harness_assert.h"
#include <cstring>  // memcmp
#include "tbb/aligned_space.h"
#include <new>      //for placement new

using std::memcmp;

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Unary minus operator applied to unsigned type, result still unsigned
    // Constant conditional expression
    #pragma warning( disable: 4127 4310 )
#endif

#if __TBB_GCC_STRICT_ALIASING_BROKEN
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

enum LoadStoreExpression {
    UseOperators,
    UseImplicitAcqRel,
    UseExplicitFullyFenced,
    UseExplicitAcqRel,
    UseExplicitRelaxed,
    UseGlobalHelperFullyFenced,
    UseGlobalHelperAcqRel,
    UseGlobalHelperRelaxed
};

//! Structure that holds an atomic<T> and some guard bytes around it.
template<typename T, LoadStoreExpression E = UseOperators>
struct TestStruct {
    typedef unsigned char byte_type;
    T prefix;
    tbb::atomic<T> counter;
    T suffix;
    TestStruct( T i ) {
        ASSERT( sizeof(*this)==3*sizeof(T), NULL );
        for (size_t j = 0; j < sizeof(T); ++j) {
            reinterpret_cast<byte_type*>(&prefix)[j]             = byte_type(0x11*(j+1));
            reinterpret_cast<byte_type*>(&suffix)[sizeof(T)-j-1] = byte_type(0x11*(j+1));
        }
        if ( E == UseOperators )
            counter = i;
        else if ( E == UseExplicitRelaxed )
            counter.template store<tbb::relaxed>(i);
        else
            tbb::store<tbb::full_fence>( counter, i );
    }
    ~TestStruct() {
        // Check for writes outside the counter.
        for (size_t j = 0; j < sizeof(T); ++j) {
            ASSERT( reinterpret_cast<byte_type*>(&prefix)[j]             == byte_type(0x11*(j+1)), NULL );
            ASSERT( reinterpret_cast<byte_type*>(&suffix)[sizeof(T)-j-1] == byte_type(0x11*(j+1)), NULL );
        }
    }
    static tbb::atomic<T> gCounter;
};

// A global variable of type tbb::atomic<>
template<typename T, LoadStoreExpression E> tbb::atomic<T> TestStruct<T, E>::gCounter;

//! Test compare_and_swap template members of class atomic<T> for memory_semantics=M
template<typename T,tbb::memory_semantics M>
void TestCompareAndSwapWithExplicitOrdering( T i, T j, T k ) {
    ASSERT( i!=k && i!=j, "values must be distinct" );
    // Test compare_and_swap that should fail
    TestStruct<T> x(i);
    T old = x.counter.template compare_and_swap<M>( j, k );
    ASSERT( old==i, NULL );
    ASSERT( x.counter==i, "old value not retained" );
    // Test compare and swap that should succeed
    old = x.counter.template compare_and_swap<M>( j, i );
    ASSERT( old==i, NULL );
    ASSERT( x.counter==j, "value not updated?" );
}

//! i, j, k must be different values
template<typename T>
void TestCompareAndSwap( T i, T j, T k ) {
    ASSERT( i!=k && i!=j, "values must be distinct" );
    // Test compare_and_swap that should fail
    TestStruct<T> x(i);
    T old = x.counter.compare_and_swap( j, k );
    ASSERT( old==i, NULL );
    ASSERT( x.counter==i, "old value not retained" );
    // Test compare and swap that should succeed
    old = x.counter.compare_and_swap( j, i );
    ASSERT( old==i, NULL );
    if( x.counter==i ) {
        ASSERT( x.counter==j, "value not updated?" );
    } else {
        ASSERT( x.counter==j, "value trashed" );
    }
    // Check that atomic global variables work
    TestStruct<T>::gCounter = i;
    old = TestStruct<T>::gCounter.compare_and_swap( j, i );
    ASSERT( old==i, NULL );
    ASSERT( TestStruct<T>::gCounter==j, "value not updated?" );
    TestCompareAndSwapWithExplicitOrdering<T,tbb::full_fence>(i,j,k);
    TestCompareAndSwapWithExplicitOrdering<T,tbb::acquire>(i,j,k);
    TestCompareAndSwapWithExplicitOrdering<T,tbb::release>(i,j,k);
    TestCompareAndSwapWithExplicitOrdering<T,tbb::relaxed>(i,j,k);
}

//! memory_semantics variation on TestFetchAndStore
template<typename T, tbb::memory_semantics M>
void TestFetchAndStoreWithExplicitOrdering( T i, T j ) {
    ASSERT( i!=j, "values must be distinct" );
    TestStruct<T> x(i);
    T old = x.counter.template fetch_and_store<M>( j );
    ASSERT( old==i, NULL );
    ASSERT( x.counter==j, NULL );
}

//! i and j must be different values
template<typename T>
void TestFetchAndStore( T i, T j ) {
    ASSERT( i!=j, "values must be distinct" );
    TestStruct<T> x(i);
    T old = x.counter.fetch_and_store( j );
    ASSERT( old==i, NULL );
    ASSERT( x.counter==j, NULL );
    // Check that atomic global variables work
    TestStruct<T>::gCounter = i;
    old = TestStruct<T>::gCounter.fetch_and_store( j );
    ASSERT( old==i, NULL );
    ASSERT( TestStruct<T>::gCounter==j, "value not updated?" );
    TestFetchAndStoreWithExplicitOrdering<T,tbb::full_fence>(i,j);
    TestFetchAndStoreWithExplicitOrdering<T,tbb::acquire>(i,j);
    TestFetchAndStoreWithExplicitOrdering<T,tbb::release>(i,j);
    TestFetchAndStoreWithExplicitOrdering<T,tbb::relaxed>(i,j);
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // conversion from <bigger integer> to <smaller integer>, possible loss of data
    // the warning seems a complete nonsense when issued for e.g. short+=short
    #pragma warning( disable: 4244 )
#endif

//! Test fetch_and_add members of class atomic<T> for memory_semantics=M
template<typename T,tbb::memory_semantics M>
void TestFetchAndAddWithExplicitOrdering( T i ) {
    TestStruct<T> x(i);
    T actual;
    T expected = i;

    // Test fetch_and_add member template
    for( int j=0; j<10; ++j ) {
        actual = x.counter.fetch_and_add(j);
        ASSERT( actual==expected, NULL );
        expected += j;
    }
    for( int j=0; j<10; ++j ) {
        actual = x.counter.fetch_and_add(-j);
        ASSERT( actual==expected, NULL );
        expected -= j;
    }

    // Test fetch_and_increment member template
    ASSERT( x.counter==i, NULL );
    actual = x.counter.template fetch_and_increment<M>();
    ASSERT( actual==i, NULL );
    ASSERT( x.counter==T(i+1), NULL );

    // Test fetch_and_decrement member template
    actual = x.counter.template fetch_and_decrement<M>();
    ASSERT( actual==T(i+1), NULL );
    ASSERT( x.counter==i, NULL );
}

//! Test fetch_and_add and related operators
template<typename T>
void TestFetchAndAdd( T i ) {
    TestStruct<T> x(i);
    T value;
    value = ++x.counter;
    ASSERT( value==T(i+1), NULL );
    value = x.counter++;
    ASSERT( value==T(i+1), NULL );
    value = x.counter--;
    ASSERT( value==T(i+2), NULL );
    value = --x.counter;
    ASSERT( value==i, NULL );
    T actual;
    T expected = i;
    for( int j=-100; j<=100; ++j ) {
        expected += j;
        actual = x.counter += j;
        ASSERT( actual==expected, NULL );
    }
    for( int j=-100; j<=100; ++j ) {
        expected -= j;
        actual = x.counter -= j;
        ASSERT( actual==expected, NULL );
    }
    // Test fetch_and_increment
    ASSERT( x.counter==i, NULL );
    actual = x.counter.fetch_and_increment();
    ASSERT( actual==i, NULL );
    ASSERT( x.counter==T(i+1), NULL );

    // Test fetch_and_decrement
    actual = x.counter.fetch_and_decrement();
    ASSERT( actual==T(i+1), NULL );
    ASSERT( x.counter==i, NULL );
    x.counter = i;
    ASSERT( x.counter==i, NULL );

    // Check that atomic global variables work
    TestStruct<T>::gCounter = i;
    value = TestStruct<T>::gCounter.fetch_and_add( 42 );
    expected = i+42;
    ASSERT( value==i, NULL );
    ASSERT( TestStruct<T>::gCounter==expected, "value not updated?" );
    TestFetchAndAddWithExplicitOrdering<T,tbb::full_fence>(i);
    TestFetchAndAddWithExplicitOrdering<T,tbb::acquire>(i);
    TestFetchAndAddWithExplicitOrdering<T,tbb::release>(i);
    TestFetchAndAddWithExplicitOrdering<T,tbb::relaxed>(i);
}

//! A type with unknown size.
class IncompleteType;

void TestFetchAndAdd( IncompleteType* ) {
    // There are no fetch-and-add operations on a IncompleteType*.
}
void TestFetchAndAdd( void* ) {
    // There are no fetch-and-add operations on a void*.
}

void TestFetchAndAdd( bool ) {
    // There are no fetch-and-add operations on a bool.
}

template<typename T>
void TestConst( T i ) {
    // Try const
    const TestStruct<T> x(i);
    ASSERT( memcmp( &i, &x.counter, sizeof(T) )==0, "write to atomic<T> broken?" );
    ASSERT( x.counter==i, "read of atomic<T> broken?" );
    const TestStruct<T, UseExplicitRelaxed> y(i);
    ASSERT( memcmp( &i, &y.counter, sizeof(T) )==0, "relaxed write to atomic<T> broken?" );
    ASSERT( tbb::load<tbb::relaxed>(y.counter) == i, "relaxed read of atomic<T> broken?" );
    const TestStruct<T, UseGlobalHelperFullyFenced> z(i);
    ASSERT( memcmp( &i, &z.counter, sizeof(T) )==0, "sequentially consistent write to atomic<T> broken?" );
    ASSERT( z.counter.template load<tbb::full_fence>() == i, "sequentially consistent read of atomic<T> broken?" );
}

#include "harness.h"

#include <sstream>

//TODO: consider moving it to separate file, and unify with one in examples command line interface
template<typename T>
std::string to_string(const T& a){
    std::stringstream str; str <<a;
    return str.str();
}
namespace initialization_tests {
    template<typename T>
    struct test_initialization_fixture{
        typedef tbb::atomic<T> atomic_t;
        tbb::aligned_space<atomic_t> non_zeroed_storage;
        enum {fill_value = 0xFF };
        test_initialization_fixture(){
            memset(non_zeroed_storage.begin(),fill_value,sizeof(non_zeroed_storage));
            ASSERT( char(fill_value)==*(reinterpret_cast<char*>(non_zeroed_storage.begin()))
                    ,"failed to fill the storage; memset error?");
        }
        //TODO: consider move it to destructor, even in a price of UB
        void tear_down(){
            non_zeroed_storage.begin()->~atomic_t();
        }
    };

    template<typename T>
    struct TestValueInitialization : test_initialization_fixture<T>{
        void operator()(){
            typedef typename test_initialization_fixture<T>::atomic_t atomic_type;
            //please note that explicit braces below are needed to get zero initialization.
            //in C++11, 8.5 Initializers [dcl.init], see  paragraphs 10,7,5
            new (this->non_zeroed_storage.begin()) atomic_type();
            //TODO: add use of KNOWN_ISSUE macro on SunCC 5.11
            #if !__SUNPRO_CC || __SUNPRO_CC > 0x5110
                //TODO: add printing of typename to the assertion
                ASSERT(char(0)==*(reinterpret_cast<char*>(this->non_zeroed_storage.begin()))
                        ,("value initialization for tbb::atomic should do zero initialization; "
                          "actual value:"+to_string(this->non_zeroed_storage.begin()->load())).c_str());
            #endif
            this->tear_down();
        };
    };

    template<typename T>
    struct TestDefaultInitialization : test_initialization_fixture<T>{
        void operator ()(){
            typedef typename test_initialization_fixture<T>::atomic_t atomic_type;
            new (this->non_zeroed_storage.begin()) atomic_type;
            ASSERT( char(this->fill_value)==*(reinterpret_cast<char*>(this->non_zeroed_storage.begin()))
                    ,"default initialization for atomic should do no initialization");
            this->tear_down();
        }
    };
#   if __TBB_ATOMIC_CTORS
        template<typename T>
        struct TestDirectInitialization : test_initialization_fixture<T> {
            void operator()(T i){
                typedef typename test_initialization_fixture<T>::atomic_t atomic_type;
                new (this->non_zeroed_storage.begin()) atomic_type(i);
                ASSERT(i == this->non_zeroed_storage.begin()->load()
                        ,("tbb::atomic initialization failed; "
                          "value:"+to_string(this->non_zeroed_storage.begin()->load())+
                          "; expected:"+to_string(i)).c_str());
                this->tear_down();
            }
        };
#   endif
}
template<typename T>
void TestValueInitialization(){
    initialization_tests::TestValueInitialization<T>()();
}
template<typename T>
void TestDefaultInitialization(){
    initialization_tests::TestDefaultInitialization<T>()();
}

#if __TBB_ATOMIC_CTORS
template<typename T>
void TestDirectInitialization(T i){
    initialization_tests::TestDirectInitialization<T>()(i);
}
//TODO:  it would be great to have constructor doing dynamic initialization of local atomic objects implicitly (with zero?),
//       but do no dynamic initializations by default for static objects
namespace test_constexpr_initialization_helper {
    struct white_box_ad_hoc_type {
        int _int;
        constexpr white_box_ad_hoc_type(int a =0) : _int(a) {};
        constexpr operator int() const { return _int; }
    };
}
//some white boxing
namespace tbb { namespace internal {
    template<>
    struct atomic_impl<test_constexpr_initialization_helper::white_box_ad_hoc_type>: atomic_impl<int> {
        atomic_impl() = default;
        constexpr atomic_impl(test_constexpr_initialization_helper::white_box_ad_hoc_type value):atomic_impl<int>(value){}
        constexpr operator int() const { return this->my_storage.my_value; }
    };
}}

//TODO: make this a parameterized macro
void TestConstExprInitializationIsTranslationTime(){
    const char* ct_init_failed_msg = "translation time init failed?";
    typedef tbb::atomic<int> atomic_t;
    constexpr atomic_t a(8);
    ASSERT(a == 8,ct_init_failed_msg);

#if !__TBB_CONSTEXPR_MEMBER_FUNCTION_BROKEN
    constexpr tbb::atomic<test_constexpr_initialization_helper::white_box_ad_hoc_type> ct_atomic(10);
    //for some unknown reason clang does not managed to enum syntax
#if __clang__
    constexpr int ct_atomic_value_ten = (int)ct_atomic;
#else
    enum {ct_atomic_value_ten = (int)ct_atomic};
#endif
    __TBB_STATIC_ASSERT(ct_atomic_value_ten == 10, "translation time init failed?");
    ASSERT(ct_atomic_value_ten == 10,ct_init_failed_msg);
    int array[ct_atomic_value_ten];
    ASSERT(Harness::array_length(array) == 10,ct_init_failed_msg);
#endif //__TBB_CONSTEXPR_MEMBER_FUNCTION_BROKEN
}

#include <string>
#include <vector>
namespace TestConstExprInitializationOfGlobalObjectsHelper{
    struct static_objects_dynamic_init_order_tester {
        static int order_hash;
        template<int N> struct nth {
            nth(){ order_hash = (order_hash<<4)+N; }
        };

        static nth<2> second;
        static nth<3> third;
    };

    int static_objects_dynamic_init_order_tester::order_hash=1;
    static_objects_dynamic_init_order_tester::nth<2> static_objects_dynamic_init_order_tester::second;
    static_objects_dynamic_init_order_tester::nth<3> static_objects_dynamic_init_order_tester::third;

    void TestStaticsDynamicInitializationOrder(){
        ASSERT(static_objects_dynamic_init_order_tester::order_hash==0x123,"Statics dynamic initialization order is broken? ");
    }

    template<typename T>
    void TestStaticInit();

    namespace auto_registered_tests_helper {
        template<typename T>
        struct type_name ;

        #define REGISTER_TYPE_NAME(T)                           \
        namespace auto_registered_tests_helper{                 \
            template<>                                          \
            struct type_name<T> {                               \
                static const char* name;                        \
            };                                                  \
            const char* type_name<T>::name = #T;                \
        }                                                       \

        typedef void (* p_test_function_type)();
        static std::vector<p_test_function_type> const_expr_tests;

        template <typename T>
        struct registration{
            registration(){const_expr_tests.push_back(&TestStaticInit<T>);}
        };
    }
    //according to ISO C++11 [basic.start.init], static data fields of class template have unordered
    //initialization unless it is an explicit specialization
    template<typename T>
    struct tester;

    #define TESTER_SPECIALIZATION(T,ct_value)                            \
    template<>                                                           \
    struct tester<T> {                                                   \
        struct static_before;                                            \
        static bool result;                                              \
        static static_before static_before_;                             \
        static tbb::atomic<T> static_atomic;                             \
                                                                         \
        static auto_registered_tests_helper::registration<T> registered; \
    };                                                                   \
    bool tester<T>::result = false;                                      \
                                                                         \
    struct tester<T>::static_before {                                    \
       static_before(){ result = (static_atomic==ct_value); }            \
    } ;                                                                  \
                                                                         \
    tester<T>::static_before tester<T>::static_before_;                  \
    tbb::atomic<T> tester<T>::static_atomic(ct_value);                   \
                                                                         \
    auto_registered_tests_helper::registration<T> tester<T>::registered; \
    REGISTER_TYPE_NAME(T)                                                \

    template<typename T>
    void TestStaticInit(){
        //TODO: add printing of values to the assertion
        std::string type_name = auto_registered_tests_helper::type_name<T>::name;
        ASSERT(tester<T>::result,("Static initialization failed for atomic " + type_name).c_str());
    }

    void CallExprInitTests(){
#   if __TBB_STATIC_CONSTEXPR_INIT_BROKEN
        REPORT("Known issue: Compile-time initialization fails for static tbb::atomic variables\n");
#   else
        using namespace auto_registered_tests_helper;
        for (size_t i =0; i<const_expr_tests.size(); ++i){
            (*const_expr_tests[i])();
        }
        REMARK("ran %d constexpr static init test \n",const_expr_tests.size());
#   endif
    }

    //TODO: unify somehow list of tested types with one in TestMain
    //TODO: add specializations for:
    //T,T(-T(1)
    //T,1
#   if __TBB_64BIT_ATOMICS
        TESTER_SPECIALIZATION(long long,8LL)
        TESTER_SPECIALIZATION(unsigned long long,8ULL)
#   endif
    TESTER_SPECIALIZATION(unsigned long,8UL)
    TESTER_SPECIALIZATION(long,8L)
    TESTER_SPECIALIZATION(unsigned int,8U)
    TESTER_SPECIALIZATION(int,8)
    TESTER_SPECIALIZATION(unsigned short,8)
    TESTER_SPECIALIZATION(short,8)
    TESTER_SPECIALIZATION(unsigned char,8)
    TESTER_SPECIALIZATION(signed char,8)
    TESTER_SPECIALIZATION(char,8)
    TESTER_SPECIALIZATION(wchar_t,8)

    int dummy;
    TESTER_SPECIALIZATION(void*,&dummy);
    TESTER_SPECIALIZATION(bool,false);
    //TODO: add test for constexpt initialization of floating types
    //for some unknown reasons 0.1 becomes 0.10000001 and equality comparison fails
    enum written_number_enum{one=2,two};
    TESTER_SPECIALIZATION(written_number_enum,one);
    //TODO: add test for ArrayElement<> as in TestMain
}

void TestConstExprInitializationOfGlobalObjects(){
    //first assert that assumption the test based on are correct
    TestConstExprInitializationOfGlobalObjectsHelper::TestStaticsDynamicInitializationOrder();
    TestConstExprInitializationOfGlobalObjectsHelper::CallExprInitTests();
}
#endif //__TBB_ATOMIC_CTORS
template<typename T>
void TestOperations( T i, T j, T k ) {
    TestValueInitialization<T>();
    TestDefaultInitialization<T>();
#   if __TBB_ATOMIC_CTORS
        TestConstExprInitializationIsTranslationTime();
        TestDirectInitialization<T>(i);
        TestDirectInitialization<T>(j);
        TestDirectInitialization<T>(k);
#   endif
    TestConst(i);
    TestCompareAndSwap(i,j,k);
    TestFetchAndStore(i,k);    // Pass i,k instead of i,j, because callee requires two distinct values.
}

template<typename T>
void TestParallel( const char* name );

bool ParallelError;

template<typename T>
struct AlignmentChecker {
    char c;
    tbb::atomic<T> i;
};

//TODO: candidate for test_compiler?
template<typename T>
void TestAlignment( const char* name ) {
    AlignmentChecker<T> ac;
    tbb::atomic<T> x;
    x = T(0);
    bool is_stack_variable_aligned = tbb::internal::is_aligned(&x,sizeof(T));
    bool is_member_variable_aligned = tbb::internal::is_aligned(&ac.i,sizeof(T));
    bool is_struct_size_correct = (sizeof(AlignmentChecker<T>)==2*sizeof(tbb::atomic<T>));
    bool known_issue_condition = __TBB_FORCE_64BIT_ALIGNMENT_BROKEN && ( sizeof(T)==8);
    //TODO: replace these ifs with KNOWN_ISSUE macro when it available
    if (!is_stack_variable_aligned){
        std::string msg = "Compiler failed to properly align local atomic variable?; size:"+to_string(sizeof(T)) + " type: "
                +to_string(name) + " location:" + to_string(&x) +"\n";
        if (known_issue_condition) {
            REPORT(("Known issue: "+ msg).c_str());
        }else{
            ASSERT(false,msg.c_str());
        }
    }
    if (!is_member_variable_aligned){
        std::string msg = "Compiler failed to properly align atomic member variable?; size:"+to_string(sizeof(T)) + " type: "
                +to_string(name) + " location:" + to_string(&ac.i) +"\n";
        if (known_issue_condition) {
            REPORT(("Known issue: "+ msg).c_str());
        }else{
            ASSERT(false,msg.c_str());
        }
    }
    if (!is_struct_size_correct){
        std::string msg = "Compiler failed to properly add padding to structure with atomic member variable?; Structure size:"+to_string(sizeof(AlignmentChecker<T>))
                + " atomic size:"+to_string(sizeof(tbb::atomic<T>)) + " type: " + to_string(name) +"\n";
        if (known_issue_condition) {
            REPORT(("Known issue: "+ msg).c_str());
        }else{
            ASSERT(false,msg.c_str());
        }
    }

    AlignmentChecker<T> array[5];
    for( int k=0; k<5; ++k ) {
        bool is_member_variable_in_array_aligned = tbb::internal::is_aligned(&array[k].i,sizeof(T));
        if (!is_member_variable_in_array_aligned) {
            std::string msg = "Compiler failed to properly align atomic member variable inside an array?; size:"+to_string(sizeof(T)) + " type:"+to_string(name)
                    + " location:" + to_string(&array[k].i) + "\n";
            if (known_issue_condition){
                REPORT(("Known issue: "+ msg).c_str());
            }else{
                ASSERT(false,msg.c_str());
            }
        }
    }
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning( disable: 4146 ) // unary minus operator applied to unsigned type, result still unsigned
    #pragma warning( disable: 4334 ) // result of 32-bit shift implicitly converted to 64 bits
#endif

/** T is an integral type. */
template<typename T>
void TestAtomicInteger( const char* name ) {
    REMARK("testing atomic<%s> (size=%d)\n",name,sizeof(tbb::atomic<T>));
    TestAlignment<T>(name);
    TestOperations<T>(0L, T(-T(1)), T(1));
    for( int k=0; k<int(sizeof(long))*8-1; ++k ) {
        const long p = 1L<<k;
        TestOperations<T>(T(p), T(~(p)), T(1-(p)));
        TestOperations<T>(T(-(p)), T(~(-(p))), T(1-(-(p))));
        TestFetchAndAdd<T>(T(-(p)));
    }
    TestParallel<T>( name );
}

namespace test_indirection_helpers {
    template<typename T>
    struct Foo {
        T x, y, z;
    };
}

template<typename T>
void TestIndirection() {
    using test_indirection_helpers::Foo;
    Foo<T> item;
    tbb::atomic<Foo<T>*> pointer;
    pointer = &item;
    for( int k=-10; k<=10; ++k ) {
        // Test various syntaxes for indirection to fields with non-zero offset.
        T value1=T(), value2=T();
        for( size_t j=0; j<sizeof(T); ++j ) {
            ((char*)&value1)[j] = char(k^j);
            ((char*)&value2)[j] = char(k^j*j);
        }
        pointer->y = value1;
        (*pointer).z = value2;
        T result1 = (*pointer).y;
        T result2 = pointer->z;
        ASSERT( memcmp(&value1,&result1,sizeof(T))==0, NULL );
        ASSERT( memcmp(&value2,&result2,sizeof(T))==0, NULL );
    }
    #if __TBB_ICC_BUILTIN_ATOMICS_POINTER_ALIASING_BROKEN
        //prevent ICC compiler from assuming 'item' is unused and reusing it's storage
        item.x = item.y=item.z;
    #endif
}

//! Test atomic<T*>
template<typename T>
void TestAtomicPointer() {
    REMARK("testing atomic pointer (%d)\n",int(sizeof(T)));
    T array[1000];
    TestOperations<T*>(&array[500],&array[250],&array[750]);
    TestFetchAndAdd<T*>(&array[500]);
    TestIndirection<T>();
    TestParallel<T*>( "pointer" );

}

//! Test atomic<Ptr> where Ptr is a pointer to a type of unknown size
template<typename Ptr>
void TestAtomicPointerToTypeOfUnknownSize( const char* name ) {
    REMARK("testing atomic<%s>\n",name);
    char array[1000];
    TestOperations<Ptr>((Ptr)(void*)&array[500],(Ptr)(void*)&array[250],(Ptr)(void*)&array[750]);
    TestParallel<Ptr>( name );
}

void TestAtomicBool() {
    REMARK("testing atomic<bool>\n");
    TestOperations<bool>(false,true,true);
    TestOperations<bool>(true,false,false);
    TestParallel<bool>( "bool" );
}

template<typename EnumType>
struct HasImplicitConversionToInt {
    typedef bool yes;
    typedef int no;
    __TBB_STATIC_ASSERT( sizeof(yes) != sizeof(no), "The helper needs two types of different sizes to work." );

    static yes detect( int );
    static no detect( ... );

    enum { value = (sizeof(yes) == sizeof(detect( EnumType() ))) };
};

enum Color {Red=0,Green=1,Blue=-1};

void TestAtomicEnum() {
    REMARK("testing atomic<Color>\n");
    TestOperations<Color>(Red,Green,Blue);
    TestParallel<Color>( "Color" );
    __TBB_STATIC_ASSERT( HasImplicitConversionToInt< tbb::atomic<Color> >::value, "The implicit conversion is expected." );
}

#if __TBB_SCOPED_ENUM_PRESENT
enum class ScopedColor1 {ScopedRed,ScopedGreen,ScopedBlue=-1};
// TODO: extend the test to cover 2 byte scoped enum as well
#if __TBB_ICC_SCOPED_ENUM_WITH_UNDERLYING_TYPE_NEGATIVE_VALUE_BROKEN
enum class ScopedColor2 : signed char {ScopedZero, ScopedOne,ScopedRed=42,ScopedGreen=-1,ScopedBlue=127};
#else
enum class ScopedColor2 : signed char {ScopedZero, ScopedOne,ScopedRed=-128,ScopedGreen=-1,ScopedBlue=127};
#endif

// TODO: replace the hack of getting symbolic enum name with a better implementation
std::string enum_strings[] = {"ScopedZero","ScopedOne","ScopedRed","ScopedGreen","ScopedBlue"};
template<>
std::string to_string<ScopedColor1>(const ScopedColor1& a){
    return enum_strings[a==ScopedColor1::ScopedBlue? 4 : (int)a+2];
}
template<>
std::string to_string<ScopedColor2>(const ScopedColor2& a){
    return enum_strings[a==ScopedColor2::ScopedRed? 2 :
        a==ScopedColor2::ScopedGreen? 3 : a==ScopedColor2::ScopedBlue? 4 : (int)a ];
}

void TestAtomicScopedEnum() {
    REMARK("testing atomic<ScopedColor>\n");
    TestOperations<ScopedColor1>(ScopedColor1::ScopedRed,ScopedColor1::ScopedGreen,ScopedColor1::ScopedBlue);
    TestParallel<ScopedColor1>( "ScopedColor1" );
#if __TBB_ICC_SCOPED_ENUM_WITH_UNDERLYING_TYPE_ATOMIC_LOAD_BROKEN
    REPORT("Known issue: the operation tests for a scoped enum with a specified underlying type are skipped.\n");
#else
    TestOperations<ScopedColor2>(ScopedColor2::ScopedRed,ScopedColor2::ScopedGreen,ScopedColor2::ScopedBlue);
    TestParallel<ScopedColor2>( "ScopedColor2" );
#endif
    __TBB_STATIC_ASSERT( !HasImplicitConversionToInt< tbb::atomic<ScopedColor1> >::value, "The implicit conversion is not expected." );
    __TBB_STATIC_ASSERT( !HasImplicitConversionToInt< tbb::atomic<ScopedColor1> >::value, "The implicit conversion is not expected." );
    __TBB_STATIC_ASSERT( sizeof(tbb::atomic<ScopedColor1>) == sizeof(ScopedColor1), "tbb::atomic instantiated with scoped enum should have the same size as scoped enum." );
    __TBB_STATIC_ASSERT( sizeof(tbb::atomic<ScopedColor2>) == sizeof(ScopedColor2), "tbb::atomic instantiated with scoped enum should have the same size as scoped enum." );
}
#endif /* __TBB_SCOPED_ENUM_PRESENT */

template<typename T>
void TestAtomicFloat( const char* name ) {
    REMARK("testing atomic<%s>\n", name );
    TestAlignment<T>(name);
    TestOperations<T>(0.5,3.25,10.75);
    TestParallel<T>( name );
}

#define __TBB_TEST_GENERIC_PART_WORD_CAS (__TBB_ENDIANNESS!=__TBB_ENDIAN_UNSUPPORTED)
#if __TBB_TEST_GENERIC_PART_WORD_CAS
void TestEndianness() {
    // Test for pure endianness (assumed by simpler probe in __TBB_MaskedCompareAndSwap()).
    bool is_big_endian = true, is_little_endian = true;
    const tbb::internal::uint32_t probe = 0x03020100;
    ASSERT (tbb::internal::is_aligned(&probe,4), NULL);
    for( const char *pc_begin = reinterpret_cast<const char*>(&probe)
         , *pc = pc_begin, *pc_end = pc_begin + sizeof(probe)
         ; pc != pc_end; ++pc) {
        if (*pc != pc_end-1-pc) is_big_endian = false;
        if (*pc != pc-pc_begin) is_little_endian = false;
    }
    ASSERT (!is_big_endian || !is_little_endian, NULL);
    #if __TBB_ENDIANNESS==__TBB_ENDIAN_DETECT
        ASSERT (is_big_endian || is_little_endian, "__TBB_ENDIANNESS should be set to __TBB_ENDIAN_UNSUPPORTED");
    #elif __TBB_ENDIANNESS==__TBB_ENDIAN_BIG
        ASSERT (is_big_endian, "__TBB_ENDIANNESS should NOT be set to __TBB_ENDIAN_BIG");
    #elif __TBB_ENDIANNESS==__TBB_ENDIAN_LITTLE
        ASSERT (is_little_endian, "__TBB_ENDIANNESS should NOT be set to __TBB_ENDIAN_LITTLE");
    #elif __TBB_ENDIANNESS==__TBB_ENDIAN_UNSUPPORTED
        #error Generic implementation of part-word CAS may not be used: unsupported endianness
    #else
        #error Unexpected value of __TBB_ENDIANNESS
    #endif
}

namespace masked_cas_helpers {
    const int numMaskedOperations = 100000;
    const int testSpaceSize = 8;
    int prime[testSpaceSize] = {3,5,7,11,13,17,19,23};

    template<typename T>
    class TestMaskedCAS_Body: NoAssign {
        T*  test_space_uncontended;
        T*  test_space_contended;
    public:
        TestMaskedCAS_Body( T* _space1, T* _space2 ) : test_space_uncontended(_space1), test_space_contended(_space2) {}
        void operator()( int my_idx ) const {
            using tbb::internal::__TBB_MaskedCompareAndSwap;
            const volatile T my_prime = T(prime[my_idx]); // 'volatile' prevents erroneous optimizations by SunCC
            T* const my_ptr = test_space_uncontended+my_idx;
            T old_value=0;
            for( int i=0; i<numMaskedOperations; ++i, old_value+=my_prime ){
                T result;
            // Test uncontended case
                T new_value = old_value + my_prime;
                // The following CAS should always fail
                result = __TBB_MaskedCompareAndSwap<T>(my_ptr,new_value,old_value-1);
                ASSERT(result!=old_value-1, "masked CAS succeeded while it should fail");
                ASSERT(result==*my_ptr, "masked CAS result mismatch with real value");
                // The following one should succeed
                result = __TBB_MaskedCompareAndSwap<T>(my_ptr,new_value,old_value);
                ASSERT(result==old_value && *my_ptr==new_value, "masked CAS failed while it should succeed");
                // The following one should fail again
                result = __TBB_MaskedCompareAndSwap<T>(my_ptr,new_value,old_value);
                ASSERT(result!=old_value, "masked CAS succeeded while it should fail");
                ASSERT(result==*my_ptr, "masked CAS result mismatch with real value");
            // Test contended case
                for( int j=0; j<testSpaceSize; ++j ){
                    // try adding my_prime until success
                    T value;
                    do {
                        value = test_space_contended[j];
                        result = __TBB_MaskedCompareAndSwap<T>(test_space_contended+j,value+my_prime,value);
                    } while( result!=value );
                }
            }
        }
    };

    template<typename T>
    struct intptr_as_array_of
    {
        static const int how_many_Ts = sizeof(intptr_t)/sizeof(T);
        union {
            intptr_t result;
            T space[ how_many_Ts ];
        };
    };

    template<typename T>
    intptr_t getCorrectUncontendedValue(int slot_idx) {
        intptr_as_array_of<T> slot;
        slot.result = 0;
        for( int i=0; i<slot.how_many_Ts; ++i ) {
            const T my_prime = T(prime[slot_idx*slot.how_many_Ts + i]);
            for( int j=0; j<numMaskedOperations; ++j )
                slot.space[i] += my_prime;
        }
        return slot.result;
    }

    template<typename T>
    intptr_t getCorrectContendedValue() {
        intptr_as_array_of<T>  slot;
        slot.result = 0;
        for( int i=0; i<slot.how_many_Ts; ++i )
            for( int primes=0; primes<testSpaceSize; ++primes )
                for( int j=0; j<numMaskedOperations; ++j )
                    slot.space[i] += prime[primes];
        return slot.result;
    }
} // namespace masked_cas_helpers

template<typename T>
void TestMaskedCAS() {
    using namespace masked_cas_helpers;
    REMARK("testing masked CAS<%d>\n",int(sizeof(T)));

    const int num_slots = sizeof(T)*testSpaceSize/sizeof(intptr_t);
    intptr_t arr1[num_slots+2]; // two more "canary" slots at boundaries
    intptr_t arr2[num_slots+2];
    for(int i=0; i<num_slots+2; ++i)
        arr2[i] = arr1[i] = 0;
    T* test_space_uncontended = (T*)(arr1+1);
    T* test_space_contended = (T*)(arr2+1);

    NativeParallelFor( testSpaceSize, TestMaskedCAS_Body<T>(test_space_uncontended, test_space_contended) );

    ASSERT( arr1[0]==0 && arr1[num_slots+1]==0 && arr2[0]==0 && arr2[num_slots+1]==0 , "adjacent memory was overwritten" );
    const intptr_t correctContendedValue = getCorrectContendedValue<T>();
    for(int i=0; i<num_slots; ++i) {
        ASSERT( arr1[i+1]==getCorrectUncontendedValue<T>(i), "unexpected value in an uncontended slot" );
        ASSERT( arr2[i+1]==correctContendedValue, "unexpected value in a contended slot" );
    }
}
#endif // __TBB_TEST_GENERIC_PART_WORD_CAS

template <typename T>
class TestRelaxedLoadStorePlainBody {
    static T s_turn,
             s_ready;

public:
    static unsigned s_count1,
                    s_count2;

    void operator() ( int id ) const {
        using tbb::internal::__TBB_load_relaxed;
        using tbb::internal::__TBB_store_relaxed;

        if ( id == 0 ) {
            while ( !__TBB_load_relaxed(s_turn) ) {
                ++s_count1;
                __TBB_store_relaxed(s_ready, 1);
            }
        }
        else {
            while ( !__TBB_load_relaxed(s_ready) ) {
                ++s_count2;
                continue;
            }
            __TBB_store_relaxed(s_turn, 1);
        }
    }
}; // class TestRelaxedLoadStorePlainBody<T>

template <typename T> T TestRelaxedLoadStorePlainBody<T>::s_turn = 0;
template <typename T> T TestRelaxedLoadStorePlainBody<T>::s_ready = 0;
template <typename T> unsigned TestRelaxedLoadStorePlainBody<T>::s_count1 = 0;
template <typename T> unsigned TestRelaxedLoadStorePlainBody<T>::s_count2 = 0;

template <typename T>
class TestRelaxedLoadStoreAtomicBody {
    static tbb::atomic<T> s_turn,
                          s_ready;

public:
    static unsigned s_count1,
                    s_count2;

    void operator() ( int id ) const {
        if ( id == 0 ) {
            while ( s_turn.template load<tbb::relaxed>() == 0 ) {
                ++s_count1;
                s_ready.template store<tbb::relaxed>(1);
            }
        }
        else {
            while ( s_ready.template load<tbb::relaxed>() == 0 ) {
                ++s_count2;
                continue;
            }
            s_turn.template store<tbb::relaxed>(1);
        }
    }
}; // class TestRelaxedLoadStoreAtomicBody<T>

template <typename T> tbb::atomic<T> TestRelaxedLoadStoreAtomicBody<T>::s_turn;
template <typename T> tbb::atomic<T> TestRelaxedLoadStoreAtomicBody<T>::s_ready;
template <typename T> unsigned TestRelaxedLoadStoreAtomicBody<T>::s_count1 = 0;
template <typename T> unsigned TestRelaxedLoadStoreAtomicBody<T>::s_count2 = 0;

template <typename T>
void TestRegisterPromotionSuppression () {
    REMARK("testing register promotion suppression (size=%d)\n", (int)sizeof(T));
    NativeParallelFor( 2, TestRelaxedLoadStorePlainBody<T>() );
    NativeParallelFor( 2, TestRelaxedLoadStoreAtomicBody<T>() );
}

template<unsigned N>
class ArrayElement {
    char item[N];
};

#include "harness_barrier.h"
namespace bit_operation_test_suite{
    struct fixture : NoAssign{
        static const uintptr_t zero = 0;
        const uintptr_t random_value ;
        const uintptr_t inverted_random_value ;
        fixture():
            random_value (tbb::internal::select_size_t_constant<0x9E3779B9,0x9E3779B97F4A7C15ULL>::value),
            inverted_random_value ( ~random_value)
        {}
    };

    struct TestAtomicORSerially : fixture {
        void operator()(){
            //these additional variable are needed to get more meaningful expression in the assert
            uintptr_t initial_value = zero;
            uintptr_t atomic_or_result = initial_value;
            uintptr_t atomic_or_operand = random_value;

            __TBB_AtomicOR(&atomic_or_result,atomic_or_operand);

            ASSERT(atomic_or_result == (initial_value | atomic_or_operand),"AtomicOR should do the OR operation");
        }
    };
    struct TestAtomicANDSerially : fixture {
        void operator()(){
            //these additional variable are needed to get more meaningful expression in the assert
            uintptr_t initial_value = inverted_random_value;
            uintptr_t atomic_and_result = initial_value;
            uintptr_t atomic_and_operand = random_value;

            __TBB_AtomicAND(&atomic_and_result,atomic_and_operand);

            ASSERT(atomic_and_result == (initial_value & atomic_and_operand),"AtomicAND should do the AND operation");
        }
    };

    struct TestAtomicORandANDConcurrently : fixture {
        static const uintptr_t bit_per_word = sizeof(uintptr_t) * 8;
        static const uintptr_t threads_number = bit_per_word;
        Harness::SpinBarrier m_barrier;
        uintptr_t bitmap;
        TestAtomicORandANDConcurrently():bitmap(zero) {}

        struct thread_body{
            TestAtomicORandANDConcurrently* test;
            thread_body(TestAtomicORandANDConcurrently* the_test) : test(the_test) {}
            void operator()(int thread_index)const{
                const uintptr_t single_bit_mask = ((uintptr_t)1u) << (thread_index % bit_per_word);
                test->m_barrier.wait();
                static const char* error_msg = "AtomicOR and AtomicAND should be atomic";
                for (uintptr_t attempts=0; attempts<1000; attempts++ ){
                    //Set and clear designated bits in a word.
                    __TBB_AtomicOR(&test->bitmap,single_bit_mask);
                     __TBB_Yield();
                    bool the_bit_is_set_after_set_via_atomic_or = ((__TBB_load_with_acquire(test->bitmap) & single_bit_mask )== single_bit_mask);
                    ASSERT(the_bit_is_set_after_set_via_atomic_or,error_msg);

                    __TBB_AtomicAND(&test->bitmap,~single_bit_mask);
                    __TBB_Yield();
                    bool the_bit_is_clear_after_clear_via_atomic_and = ((__TBB_load_with_acquire(test->bitmap) & single_bit_mask )== zero);
                    ASSERT(the_bit_is_clear_after_clear_via_atomic_and,error_msg);
                }
            }
        };
        void operator()(){
            m_barrier.initialize(threads_number);
            NativeParallelFor(threads_number,thread_body(this));
        }
    };
}
void TestBitOperations(){
    using namespace bit_operation_test_suite;
    TestAtomicORSerially()();
    TestAtomicANDSerially()();
    TestAtomicORandANDConcurrently()();
}

int TestMain () {
#   if __TBB_ATOMIC_CTORS
         TestConstExprInitializationOfGlobalObjects();
#   endif //__TBB_ATOMIC_CTORS
#   if __TBB_64BIT_ATOMICS && !__TBB_CAS_8_CODEGEN_BROKEN
         TestAtomicInteger<unsigned long long>("unsigned long long");
         TestAtomicInteger<long long>("long long");
#   elif __TBB_CAS_8_CODEGEN_BROKEN
         REPORT("Known issue: compiler generates incorrect code for 64-bit atomics on this configuration\n");
#   else
         REPORT("Known issue: 64-bit atomics are not supported\n");
         ASSERT(sizeof(long long)==8, "type long long is not 64 bits");
#   endif
    TestAtomicInteger<unsigned long>("unsigned long");
    TestAtomicInteger<long>("long");
    TestAtomicInteger<unsigned int>("unsigned int");
    TestAtomicInteger<int>("int");
    TestAtomicInteger<unsigned short>("unsigned short");
    TestAtomicInteger<short>("short");
    TestAtomicInteger<signed char>("signed char");
    TestAtomicInteger<unsigned char>("unsigned char");
    TestAtomicInteger<char>("char");
    TestAtomicInteger<wchar_t>("wchar_t");
    TestAtomicInteger<size_t>("size_t");
    TestAtomicInteger<ptrdiff_t>("ptrdiff_t");
    TestAtomicPointer<ArrayElement<1> >();
    TestAtomicPointer<ArrayElement<2> >();
    TestAtomicPointer<ArrayElement<3> >();
    TestAtomicPointer<ArrayElement<4> >();
    TestAtomicPointer<ArrayElement<5> >();
    TestAtomicPointer<ArrayElement<6> >();
    TestAtomicPointer<ArrayElement<7> >();
    TestAtomicPointer<ArrayElement<8> >();
    TestAtomicPointerToTypeOfUnknownSize<IncompleteType*>( "IncompleteType*" );
    TestAtomicPointerToTypeOfUnknownSize<void*>( "void*" );
    TestAtomicBool();
    TestAtomicEnum();
#   if __TBB_SCOPED_ENUM_PRESENT
    TestAtomicScopedEnum();
#   endif
    TestAtomicFloat<float>("float");
#   if __TBB_64BIT_ATOMICS && !__TBB_CAS_8_CODEGEN_BROKEN
        TestAtomicFloat<double>("double");
#   else
        ASSERT(sizeof(double)==8, "type double is not 64 bits");
#   endif
    ASSERT( !ParallelError, NULL );
#   if __TBB_TEST_GENERIC_PART_WORD_CAS
        TestEndianness();
        ASSERT (sizeof(short)==2, NULL);
        TestMaskedCAS<unsigned short>();
        TestMaskedCAS<short>();
        TestMaskedCAS<unsigned char>();
        TestMaskedCAS<signed char>();
        TestMaskedCAS<char>();
#   elif __TBB_USE_GENERIC_PART_WORD_CAS
#       error Generic part-word CAS is enabled, but not covered by the test
#   else
        REPORT("Skipping test for generic part-word CAS\n");
#   endif
#   if __TBB_64BIT_ATOMICS && !__TBB_CAS_8_CODEGEN_BROKEN
        TestRegisterPromotionSuppression<tbb::internal::int64_t>();
#   endif
    TestRegisterPromotionSuppression<tbb::internal::int32_t>();
    TestRegisterPromotionSuppression<tbb::internal::int16_t>();
    TestRegisterPromotionSuppression<tbb::internal::int8_t>();
    TestBitOperations();

    return Harness::Done;
}

template<typename T, bool aligned>
class AlignedAtomic: NoAssign {
    //tbb::aligned_space can not be used here, because internally it utilize align pragma/attribute,
    //which has bugs on 8byte alignment on ia32 on some compilers( see according ****_BROKEN macro)
    // Allocate space big enough to always contain sizeof(T)-byte locations that are aligned and misaligned.
    char raw_space[2*sizeof(T) -1];
public:
    tbb::atomic<T>& construct_atomic(){
        std::memset(&raw_space[0],0, sizeof(raw_space));
        uintptr_t delta = aligned ? 0 : sizeof(T)/2;
        size_t index=sizeof(T)-1;
        tbb::atomic<T>* y = reinterpret_cast<tbb::atomic<T>*>((reinterpret_cast<uintptr_t>(&raw_space[index+delta])&~index) - delta);
        // Assertion checks that y really did end up somewhere inside "raw_space".
        ASSERT( raw_space<=reinterpret_cast<char*>(y), "y starts before raw_space" );
        ASSERT( reinterpret_cast<char*>(y+1) <= raw_space+sizeof(raw_space), "y starts after raw_space" );
        ASSERT( !(aligned ^ tbb::internal::is_aligned(y,sizeof(T))), "y is not aligned as it required" );
        new (y) tbb::atomic<T> ();
        return *y;
    }
};

template<typename T, bool aligned>
struct FlagAndMessage: AlignedAtomic<T,aligned> {
    //! 0 if message not set yet, 1 if message is set.
    tbb::atomic<T>& flag;
    /** Force flag and message to be on distinct cache lines for machines with cache line size <= 4096 bytes */
    char pad[4096/sizeof(T)];
    //! Non-zero if message is ready
    T message;
    FlagAndMessage(): flag(FlagAndMessage::construct_atomic()) {
        std::memset(pad,0,sizeof(pad));
    }
};

// A special template function used for summation.
// Actually it is only necessary because of its specialization for void*
template<typename T>
T special_sum(intptr_t arg1, intptr_t arg2) {
    return (T)((T)arg1 + arg2);
}

// The specialization for IncompleteType* is required
// because pointer arithmetic (+) is impossible with IncompleteType*
template<>
IncompleteType* special_sum<IncompleteType*>(intptr_t arg1, intptr_t arg2) {
    return (IncompleteType*)(arg1 + arg2);
}

// The specialization for void* is required
// because pointer arithmetic (+) is impossible with void*
template<>
void* special_sum<void*>(intptr_t arg1, intptr_t arg2) {
    return (void*)(arg1 + arg2);
}

// The specialization for bool is required to shut up gratuitous compiler warnings,
// because some compilers warn about casting int to bool.
template<>
bool special_sum<bool>(intptr_t arg1, intptr_t arg2) {
    return ((arg1!=0) + arg2)!=0;
}

#if __TBB_SCOPED_ENUM_PRESENT
// The specialization for scoped enumerators is required
// because scoped enumerators prohibit implicit conversion to int
template<>
ScopedColor1 special_sum<ScopedColor1>(intptr_t arg1, intptr_t arg2) {
    return (ScopedColor1)(arg1 + arg2);
}
template<>
ScopedColor2 special_sum<ScopedColor2>(intptr_t arg1, intptr_t arg2) {
    return (ScopedColor2)(arg1 + arg2);
}
#endif

volatile int One = 1;

inline bool IsRelaxed ( LoadStoreExpression e ) {
    return e == UseExplicitRelaxed || e == UseGlobalHelperRelaxed;
}

template <typename T, LoadStoreExpression E>
struct LoadStoreTraits;

template <typename T>
struct LoadStoreTraits<T, UseOperators> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = src; }
    static void store ( tbb::atomic<T>& dst, const T& src ) { dst = src; }
};

template <typename T>
struct LoadStoreTraits<T, UseImplicitAcqRel> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = src.load(); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { dst.store(src); }
};

template <typename T>
struct LoadStoreTraits<T, UseExplicitFullyFenced> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = src.template load<tbb::full_fence>(); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { dst.template store<tbb::full_fence>(src); }
};

template <typename T>
struct LoadStoreTraits<T, UseExplicitAcqRel> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = src.template load<tbb::acquire>(); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { dst.template store<tbb::release>(src); }
};

template <typename T>
struct LoadStoreTraits<T, UseExplicitRelaxed> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = src.template load<tbb::relaxed>(); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { dst.template store<tbb::relaxed>(src); }
};

template <typename T>
struct LoadStoreTraits<T, UseGlobalHelperFullyFenced> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = tbb::load<tbb::full_fence>(src); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { tbb::store<tbb::full_fence>(dst, src); }
};

template <typename T>
struct LoadStoreTraits<T, UseGlobalHelperAcqRel> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = tbb::load<tbb::acquire>(src); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { tbb::store<tbb::release>(dst, src); }
};

template <typename T>
struct LoadStoreTraits<T, UseGlobalHelperRelaxed> {
    static void load ( T& dst, const tbb::atomic<T>& src ) { dst = tbb::load<tbb::relaxed>(src); }
    static void store ( tbb::atomic<T>& dst, const T& src ) { tbb::store<tbb::relaxed>(dst, src); }
};

template<typename T, bool aligned, LoadStoreExpression E>
struct HammerLoadAndStoreFence: NoAssign {
    typedef FlagAndMessage<T,aligned> fam_type;
private:
    typedef LoadStoreTraits<T, E> trait;
    fam_type* fam;
    const int n;
    const int p;
    const int trial;
    const char* name;
    mutable T accum;
public:
    HammerLoadAndStoreFence( fam_type* fam_, int n_, int p_, const char* name_, int trial_ ) : fam(fam_), n(n_), p(p_), trial(trial_), name(name_) {}
    void operator()( int k ) const {
        int one = One;
        fam_type* s = fam+k;
        fam_type* s_next = fam + (k+1)%p;
        for( int i=0; i<n; ++i ) {
            // The inner for loop is a spin-wait loop, which is normally considered very bad style.
            // But we must use it here because we are interested in examining subtle hardware effects.
            for(unsigned short cnt=1; ; ++cnt) {
                if( !(cnt%1024) ) // to help 1-core or oversubscribed systems complete the test, yield every 2^10 iterations
                    __TBB_Yield();
                // Compilers typically generate non-trivial sequence for division by a constant.
                // The expression here is dependent on the loop index i, so it cannot be hoisted.
                #define COMPLICATED_ZERO (i*(one-1)/100)
                // Read flag and then the message
                T flag, message;
                if( trial&1 ) {
                    // COMPLICATED_ZERO here tempts compiler to hoist load of message above reading of flag.
                    trait::load( flag, (s+COMPLICATED_ZERO)->flag );
                    message = s->message;
                } else {
                    trait::load( flag, s->flag );
                    message = s->message;
                }
                if ( flag != T(0) ) {
                    if( flag!=(T)-1 ) {
                        REPORT("ERROR: flag!=(T)-1 k=%d i=%d trial=%x type=%s (atomicity problem?)\n", k, i, trial, name );
                        ParallelError = true;
                    }
                    if( !IsRelaxed(E) && message!=(T)-1 ) {
                        REPORT("ERROR: message!=(T)-1 k=%d i=%d trial=%x type=%s mode=%d (memory fence problem?)\n", k, i, trial, name, E );
                        ParallelError = true;
                    }
                    s->message = T(0);
                    trait::store( s->flag, T(0) );
                    // Prevent deadlock possible in relaxed mode because of store(0)
                    // to the first thread's flag being reordered after the last
                    // thread's store(-1) into it.
                    if ( IsRelaxed(E) ) {
                        while( s_next->flag.template load<tbb::relaxed>() != T(0) )
                            __TBB_Yield();
                    }
                    else
                        ASSERT( s_next->flag == T(0), NULL );
                    // Set message and then the flag
                    if( trial&2 ) {
                        // COMPLICATED_ZERO here tempts compiler to sink store below setting of flag
                        s_next->message = special_sum<T>(-1, COMPLICATED_ZERO);
                        trait::store( s_next->flag, (T)-1 );
                    } else {
                        s_next->message = (T)-1;
                        trait::store( s_next->flag, (T)-1 );
                    }
                    break;
                } else {
                    // Force compiler to use message anyway, so it cannot sink read of s->message below the if.
                    accum = message;
                }
            }
        }
    }
};

//! Test that atomic<T> has acquire semantics for loads and release semantics for stores.
/** Test performs round-robin passing of message among p processors,
    where p goes from MinThread to MaxThread. */
template<typename T, bool aligned, LoadStoreExpression E>
void TestLoadAndStoreFences( const char* name ) {
    typedef HammerLoadAndStoreFence<T, aligned, E> hammer_load_store_type;
    typedef typename hammer_load_store_type::fam_type fam_type;
    for( int p=MinThread<2 ? 2 : MinThread; p<=MaxThread; ++p ) {
        fam_type * fam = new fam_type[p];
        // Each of four trials exercise slightly different expression pattern within the test.
        // See occurrences of COMPLICATED_ZERO for details.
        for( int trial=0; trial<4; ++trial ) {
            fam->message = (T)-1;
            fam->flag = (T)-1;
            NativeParallelFor( p, hammer_load_store_type( fam, 100, p, name, trial ) );
            if ( !IsRelaxed(E) ) {
                for( int k=0; k<p; ++k ) {
                    ASSERT( fam[k].message==(k==0 ? (T)-1 : T(0)), "incomplete round-robin?" );
                    ASSERT( fam[k].flag==(k==0 ? (T)-1 : T(0)), "incomplete round-robin?" );
                }
            }
        }
        delete[] fam;
    }
}

//! Sparse set of values of integral type T.
/** Set is designed so that if a value is read or written non-atomically,
    the resulting intermediate value is likely to not be a member of the set. */
template<typename T>
class SparseValueSet {
    T factor;
public:
    SparseValueSet() {
        // Compute factor such that:
        // 1. It has at least one 1 in most of its bytes.
        // 2. The bytes are typically different.
        // 3. When multiplied by any value <=127, the product does not overflow.
        factor = T(0);
        for( unsigned i=0; i<sizeof(T)*8-7; i+=7 )
            factor = T(factor | T(1)<<i);
     }
     //! Get ith member of set
     T get( int i ) const {
         // Create multiple of factor.  The & prevents overflow of the product.
         return T((i&0x7F)*factor);
     }
     //! True if set contains x
     bool contains( T x ) const {
         // True if
         return (x%factor)==0;
     }
};

//! Specialization for pointer types.  The pointers are random and should not be dereferenced.
template<typename T>
class SparseValueSet<T*> {
    SparseValueSet<ptrdiff_t> my_set;
public:
    T* get( int i ) const {return reinterpret_cast<T*>(my_set.get(i));}
    bool contains( T* x ) const {return my_set.contains(reinterpret_cast<ptrdiff_t>(x));}
};

//! Specialization for bool.
/** Checking bool for atomic read/write is pointless in practice, because
    there is no way to *not* atomically read or write a bool value. */
template<>
class SparseValueSet<bool> {
public:
    bool get( int i ) const {return i&1;}
    bool contains( bool ) const {return true;}
};

#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
    // VS2008/VC9 seems to have an issue; limits pull in math.h
    #pragma warning( push )
    #pragma warning( disable: 4985 )
#endif
#include <limits> /* Need std::numeric_limits */
#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
    #pragma warning( pop )
#endif

//! Commonality inherited by specializations for floating-point types.
template<typename T>
class SparseFloatSet: NoAssign {
    const T epsilon;
public:
    SparseFloatSet() : epsilon(std::numeric_limits<T>::epsilon()) {}
    T get( int i ) const {
        return i==0 ? T(0) : 1/T((i&0x7F)+1);
    }
    bool contains( T x ) const {
        if( x==T(0) ) {
            return true;
        } else {
            int j = int(1/x+T(0.5));
            if( 0<j && j<=128 ) {
                T error = x*T(j)-T(1);
                // In the calculation above, if x was indeed generated by method get, the error should be
                // at most epsilon, because x is off by at most 1/2 ulp from its infinitely precise value,
                // j is exact, and the multiplication incurs at most another 1/2 ulp of round-off error.
                if( -epsilon<=error && error<=epsilon ) {
                    return true;
                } else {
                    REPORT("Warning: excessive floating-point error encountered j=%d x=%.15g error=%.15g\n",j,x,error);
                }
            }
            return false;
        }
    };
};

template<>
class SparseValueSet<float>: public SparseFloatSet<float> {};

template<>
class SparseValueSet<double>: public SparseFloatSet<double> {};

#if __TBB_SCOPED_ENUM_PRESENT
//! Commonality inherited by specializations for scoped enumerator types.
template<typename EnumType>
class SparseEnumValueSet {
public:
    EnumType get( int i ) const {return i%3==0 ? EnumType::ScopedRed : i%3==1 ? EnumType::ScopedGreen : EnumType::ScopedBlue;}
    bool contains( EnumType e ) const {return e==EnumType::ScopedRed || e==EnumType::ScopedGreen || e==EnumType::ScopedBlue;}
};
template<>
class SparseValueSet<ScopedColor1> : public SparseEnumValueSet<ScopedColor1> {};
template<>
class SparseValueSet<ScopedColor2> : public SparseEnumValueSet<ScopedColor2> {};
#endif

template<typename T, bool aligned>
class HammerAssignment: AlignedAtomic<T,aligned> {
    tbb::atomic<T>& x;
    const char* name;
    SparseValueSet<T> set;
public:
    HammerAssignment(const char* name_ ) : x(HammerAssignment::construct_atomic()), name(name_) {
        x = set.get(0);
    }
    void operator()( int k ) const {
        const int n = 1000000;
        if( k ) {
            tbb::atomic<T> z;
            AssertSameType( z=x, z );    // Check that return type from assignment is correct
            for( int i=0; i<n; ++i ) {
                // Read x atomically into z.
                z = x;
                if( !set.contains(z) ) {
                    REPORT("ERROR: assignment of atomic<%s> is not atomic\n", name);
                    ParallelError = true;
                    return;
                }
            }
        } else {
            tbb::atomic<T> y;
            for( int i=0; i<n; ++i ) {
                // Get pseudo-random value.
                y = set.get(i);
                // Write y atomically into x.
                x = y;
            }
        }
    }
};

// Compile-time check that a class method has the required signature.
// Intended to check the assignment operator of tbb::atomic.
template<typename T> void TestAssignmentSignature( T& (T::*)(const T&) ) {}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning( disable: 4355 4800 )
#endif

template<typename T, bool aligned>
void TestAssignment( const char* name ) {
    TestAssignmentSignature( &tbb::atomic<T>::operator= );
    NativeParallelFor( 2, HammerAssignment<T,aligned>(name ) );
}

template <typename T, bool aligned, LoadStoreExpression E>
class DekkerArbitrationBody : NoAssign, Harness::NoAfterlife {
    typedef LoadStoreTraits<T, E> trait;

    mutable Harness::FastRandom my_rand;
    static const unsigned short c_rand_ceil = 10;
    mutable AlignedAtomic<T,aligned> s_ready_storage[2];
    mutable AlignedAtomic<T,aligned> s_turn_storage;
    mutable tbb::atomic<T>* s_ready[2];
    tbb::atomic<T>& s_turn;
    mutable volatile bool s_inside;

public:
    void operator() ( int id ) const {
        const int me = id;
        const T other = (T)(uintptr_t)(1 - id),
                cleared = T(0),
                signaled = T(1);
        for ( int i = 0; i < 100000; ++i ) {
            trait::store( *s_ready[me], signaled );
            trait::store( s_turn, other );
            T r, t;
            for ( int j = 0; ; ++j ) {
                trait::load(r, *s_ready[(uintptr_t)other]);
                trait::load(t, s_turn);
                if ( r != signaled || t != other )
                    break;
                __TBB_Pause(1);
                if ( j == 2<<12 ) {
                    j = 0;
                    __TBB_Yield();
                }
            }
            // Entered critical section
            ASSERT( !s_inside, "Peterson lock is broken - some fences are missing" );
            s_inside = true;
            unsigned short spin = my_rand.get() % c_rand_ceil;
            for ( volatile int j = 0; j < spin; ++j )
                continue;
            s_inside = false;
            ASSERT( !s_inside, "Peterson lock is broken - some fences are missing" );
            // leaving critical section
            trait::store( *s_ready[me], cleared );
            spin = my_rand.get() % c_rand_ceil;
            for ( volatile int j = 0; j < spin; ++j )
                continue;
        }
    }

    DekkerArbitrationBody ()
        : my_rand((unsigned)(uintptr_t)this)
        , s_turn(s_turn_storage.construct_atomic())
        , s_inside (false)
    {
        //atomics pointed to by s_ready and s_turn will be zeroed by the
        //according construct_atomic() calls
         s_ready[0] = &s_ready_storage[0].construct_atomic();
         s_ready[1] = &s_ready_storage[1].construct_atomic();
    }
};

template <typename T, bool aligned, LoadStoreExpression E>
void TestDekkerArbitration () {
    NativeParallelFor( 2, DekkerArbitrationBody<T,aligned, E>() );
}

template<typename T>
void TestParallel( const char* name ) {
    //TODO: looks like there are no tests for operations other than load/store ?
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
    if (sizeof(T)==8){
        TestLoadAndStoreFences<T, false, UseOperators>(name);
        TestLoadAndStoreFences<T, false, UseImplicitAcqRel>(name);
        TestLoadAndStoreFences<T, false, UseExplicitFullyFenced>(name);
        TestLoadAndStoreFences<T, false, UseExplicitAcqRel>(name);
        TestLoadAndStoreFences<T, false, UseExplicitRelaxed>(name);
        TestLoadAndStoreFences<T, false, UseGlobalHelperFullyFenced>(name);
        TestLoadAndStoreFences<T, false, UseGlobalHelperAcqRel>(name);
        TestLoadAndStoreFences<T, false, UseGlobalHelperRelaxed>(name);
        TestAssignment<T,false>(name);
        TestDekkerArbitration<T, false, UseExplicitFullyFenced>();
        TestDekkerArbitration<T, false, UseGlobalHelperFullyFenced>();
    }
#endif

    TestLoadAndStoreFences<T, true, UseOperators>(name);
    TestLoadAndStoreFences<T, true, UseImplicitAcqRel>(name);
    TestLoadAndStoreFences<T, true, UseExplicitFullyFenced>(name);
    TestLoadAndStoreFences<T, true, UseExplicitAcqRel>(name);
    TestLoadAndStoreFences<T, true, UseExplicitRelaxed>(name);
    TestLoadAndStoreFences<T, true, UseGlobalHelperFullyFenced>(name);
    TestLoadAndStoreFences<T, true, UseGlobalHelperAcqRel>(name);
    TestLoadAndStoreFences<T, true, UseGlobalHelperRelaxed>(name);
    TestAssignment<T,true>(name);
    TestDekkerArbitration<T, true, UseExplicitFullyFenced>();
    TestDekkerArbitration<T, true, UseGlobalHelperFullyFenced>();
}

#endif // __TBB_TEST_SKIP_PIC_MODE || __TBB_TEST_SKIP_BUILTINS_MODE
