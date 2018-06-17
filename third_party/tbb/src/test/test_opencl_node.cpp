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

#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

#include "tbb/tbb_config.h"

// The old versions of MSVC 2013 fail to compile the test with fatal error
#define __TBB_MSVC_TEST_COMPILATION_BROKEN (_MSC_VER && _MSC_FULL_VER <= 180021005 && !__INTEL_COMPILER)

#if __TBB_PREVIEW_OPENCL_NODE && !__TBB_MSVC_TEST_COMPILATION_BROKEN
#if _MSC_VER
#pragma warning (disable: 4503) // Suppress "decorated name length exceeded, name was truncated" warning
#endif
#include <iterator>
#include "tbb/task_scheduler_init.h"
#include <vector>
#include <iostream>

#include "tbb/flow_graph_opencl_node.h"
using namespace tbb::flow;

#include "harness_assert.h"

#if ( __INTEL_COMPILER && __INTEL_COMPILER <= 1500 ) || __clang__
// In some corner cases the compiler fails to perform automatic type deduction for function pointer.
// Workaround is to replace a function pointer with a function call.
#define BROKEN_FUNCTION_POINTER_DEDUCTION(...) __VA_ARGS__()
#else
#define BROKEN_FUNCTION_POINTER_DEDUCTION(...) __VA_ARGS__
#endif

#if _MSC_VER <= 1800 && !__INTEL_COMPILER
// In some corner cases the compiler fails to perform automatic std::initializer_list deduction for curly brackets.
// Workaround is to perform implicit conversion.
template <typename T>
std::initializer_list<T> make_initializer_list( std::initializer_list<T> il ) { return il; }
#define BROKEN_INITIALIZER_LIST_DEDUCTION(...) make_initializer_list(__VA_ARGS__)
#else
#define BROKEN_INITIALIZER_LIST_DEDUCTION(...) __VA_ARGS__
#endif

#include "harness.h"

#include <mutex>
std::once_flag tbbRootFlag;
char *tbbRoot = NULL;
std::string PathToFile(const std::string& fileName) {
    std::call_once(tbbRootFlag, [] { tbbRoot = Harness::GetEnv("tbb_root"); });
    std::string prefix = tbbRoot ? tbbRoot : "../..";
    return prefix + "/src/test/" + fileName;
}

// Global test variables and types
typedef tbb::flow::opencl_range OCLRange;
struct test_default_device_filter {
    opencl_device_list operator()(const opencl_device_list &devices) {
        opencl_device_list dl;
        dl.add(*devices.begin());
        return dl;
    }
};
typedef opencl_factory<test_default_device_filter> DefaultFactoryType;

void TestArgumentPassing() {
    REMARK( "TestArgumentPassing: " );

    graph g;
    opencl_node <tuple<opencl_buffer<int>, opencl_buffer<int>, OCLRange>> k( g,
        opencl_program<>( PathToFile( "test_opencl_node.cl" ) ).get_kernel( "TestArgumentPassing" ) );
    split_node <tuple<opencl_buffer<int>, opencl_buffer<int>, OCLRange>> s( g );

    make_edge( output_port<0>( s ), input_port<0>( k ) );
    make_edge( output_port<1>( s ), input_port<1>( k ) );
    make_edge( output_port<2>( s ), input_port<2>( k ) );

    const int N = 1 * 1024 * 1024;
    opencl_buffer<int> b1( N ), b2( N );

    const int err_size = 128;
    opencl_buffer<char> err( err_size );

    OCLRange l;

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 1 );
    k.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }), BROKEN_INITIALIZER_LIST_DEDUCTION({ 16 }) } );
    k.set_args( port_ref<0, 1>(), /* stride_x */ 1, /* stride_y */ 0, /* stride_z */ 0, /* dim */ 1, err, err_size );
    s.try_put( std::tie( b1, b2, l ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin(), b2.end(), []( int c ) { return c == 1; } ), "Validation has failed" );

    // By default, the first device is used.
    opencl_device d = *interface10::opencl_info::available_devices().begin();
    std::array<size_t, 3> maxSizes = d.max_work_item_sizes();

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 2 );
    int stride_x = 1;
    k.set_args( port_ref<0>(), BROKEN_FUNCTION_POINTER_DEDUCTION( port_ref<1, 1> ), stride_x, /* stride_y */ 1024, /* stride_z */ 0, /* dim */ 2, err, err_size );
    k.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ 1024, 1024 }),
                   BROKEN_INITIALIZER_LIST_DEDUCTION({ 16, min( (int)maxSizes[1], 16 ) }) } );
    s.try_put( std::tie( b1, b2, l ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin(), b2.end(), []( int c ) { return c == 2; } ), "Validation has failed" );

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 3 );
    stride_x = 2; // Nothing should be changed
    s.try_put( std::tie( b1, b2, l ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin(), b2.end(), []( int c ) { return c == 3; } ), "Validation has failed" );

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 4 );
    int stride_z = 64 * 64;
    ASSERT( stride_z * 64 < N, NULL );
    k.set_args( port_ref<0>(), BROKEN_FUNCTION_POINTER_DEDUCTION( port_ref<1> ), /* stride_x */ 1, /* stride_y */ 64, /* stride_z */ stride_z, /* dim */ 3, err, err_size );
    k.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ 64, 64, 64 }),
                   BROKEN_INITIALIZER_LIST_DEDUCTION({ 4, min( (int)maxSizes[1], 4 ), min( (int)maxSizes[2], 4 ) }) } );
    s.try_put( std::make_tuple( b1, b2, OCLRange() ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin(), b2.begin() + stride_z * 64, []( int c ) { return c == 4; } ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin() + stride_z * 64, b2.end(), []( int c ) { return c == 3; } ), "Validation has failed" );

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 5 );
    ASSERT( 2 * 64 * 64 < N, NULL );
    k.set_args( port_ref<0, 1>(), /* stride_x */ 2, /* stride_y */ 2 * 64, /* stride_z */ 2 * 64 * 64, /* dim */ 3, err, err_size );
    k.set_range( BROKEN_FUNCTION_POINTER_DEDUCTION( port_ref<2> ) );
    l = { BROKEN_INITIALIZER_LIST_DEDUCTION({ 64, 64, 64 }),
          BROKEN_INITIALIZER_LIST_DEDUCTION({ 4, min( (int)maxSizes[1], 4 ), min( (int)maxSizes[2], 4 ) }) };
    s.try_put( std::make_tuple( b1, b2, l ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    auto it = b2.begin();
    for ( size_t i = 0; i < 64 * 64 * 64; ++i ) ASSERT( it[i] == (i % 2 ? 4 : 5), "Validation has failed" );
    for ( size_t i = 64 * 64 * 64; i < 2 * 64 * 64 * 64; ++i ) ASSERT( it[i] == (i % 2 ? 3 : 5), "Validation has failed" );
    ASSERT( std::all_of( b2.begin() + 2 * stride_z * 64, b2.end(), []( int c ) { return c == 3; } ), "Validation has failed" );

    *err.data() = 0; ASSERT( err.data() != std::string( "Done" ), NULL );
    std::fill( b1.begin(), b1.end(), 6 );
    k.set_args( port_ref<0, 1>(), /* stride_x */ 1, /* stride_y */ 1024, /* stride_z */ 0, /* dim */ 2, err, err_size );
    k.set_range( std::deque<int>( { 1024, 1024 } ) );
    s.try_put( std::make_tuple( b1, b2, l ) );
    g.wait_for_all();
    ASSERT( err.data() == std::string( "Done" ), "Validation has failed" );
    ASSERT( std::all_of( b2.begin(), b2.end(), []( int c ) { return c == 6; } ), "Validation has failed" );
    REMARK( "done\n" );
}

void SimpleDependencyTest() {
    REMARK( "SimpleDependencyTest: " );

    const int N = 1 * 1024 * 1024;
    opencl_buffer<float> b1( N ), b2( N ), b3( N );
    std::vector<float> v1( N ), v2( N ), v3( N );

    auto i1 = b1.access<write_only>();
    auto i2 = b2.access<write_only>();

    for ( int i = 0; i < N; ++i ) {
        i1[i] = v1[i] = float( i );
        i2[i] = v2[i] = float( 2 * i );
    }

    graph g;
    opencl_program<> p( PathToFile("test_opencl_node.cl") ) ;
    opencl_node< tuple<opencl_buffer<float>, opencl_buffer<float>> > k1( g, p.get_kernel( "Sum" ) );
    k1.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }), BROKEN_INITIALIZER_LIST_DEDUCTION({ 16 }) } );

    opencl_node < tuple<opencl_buffer<float>, opencl_buffer<float>> > k2( g, p.get_kernel( "Sqr" ) );
    k2.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }), BROKEN_INITIALIZER_LIST_DEDUCTION({ 16 }) } );

    make_edge( output_port<1>( k1 ), input_port<0>( k2 ) );

    split_node< tuple<opencl_buffer<float>, opencl_buffer<float>, opencl_buffer<float>> > s( g );

    make_edge( output_port<0>( s ), input_port<0>( k1 ) );
    make_edge( output_port<1>( s ), input_port<1>( k1 ) );
    make_edge( output_port<2>( s ), input_port<1>( k2 ) );

    s.try_put( std::tie( b1, b2, b3 ) );

    g.wait_for_all();

    // validation
    for ( int i = 0; i < N; ++i ) {
        v2[i] += v1[i];
        v3[i] = v2[i] * v2[i];
    }

    auto o2 = b2.access<read_only>();
    auto o3 = b3.access<read_only>();

    ASSERT( memcmp( &o2[0], &v2[0], N*sizeof( float ) ) == 0, "Validation has failed" );
    ASSERT( memcmp( &o3[0], &v3[0], N*sizeof( float ) ) == 0, "Validation has failed" );
    REMARK( "done\n" );
}

class device_selector {
    enum state {
        DEFAULT_INITIALIZED,
        COPY_INITIALIZED,
        DELETED
    };
    state my_state;
public:
    device_selector() : my_state( DEFAULT_INITIALIZED ) {}
    device_selector( const device_selector& ) : my_state( COPY_INITIALIZED ) {}
    device_selector( device_selector&& ) : my_state( COPY_INITIALIZED ) {}
    ~device_selector() { my_state = DELETED; }

    template <typename D>
    opencl_device operator()( opencl_factory<D> &f ) {
        ASSERT( my_state == COPY_INITIALIZED, NULL );
        ASSERT( ! f.devices().empty(), NULL );
        return *( f.devices().begin() );
    }
};

void BroadcastTest() {
    REMARK( "BroadcastTest: " );

    graph g;

    const int N = 1 * 1024;
    opencl_buffer<cl_int> b( N );

    const int numNodes = 4 * tbb::task_scheduler_init::default_num_threads();
    typedef opencl_node <tuple<opencl_buffer<cl_int>, opencl_buffer<cl_int>>> NodeType;
    std::vector<NodeType> nodes( numNodes, NodeType( g,
        opencl_program<>( PathToFile("test_opencl_node.cl") ).get_kernel( "BroadcastTest" ),
        device_selector() ) );

    for ( std::vector<NodeType>::iterator it = nodes.begin(); it != nodes.end(); ++it )
        it->set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }), BROKEN_INITIALIZER_LIST_DEDUCTION({ 16 }) } );

    broadcast_node<opencl_buffer<cl_int>> bc( g );
    for ( auto &x : nodes ) make_edge( bc, x );

    std::vector<opencl_buffer<cl_int>> res;
    for ( int i = 0; i < numNodes; ++i ) res.emplace_back( N );

    for ( cl_int r = 1; r < 100; ++r ) {
        std::fill( b.begin(), b.end(), r );
        bc.try_put( b );
        for ( int i = 0; i < numNodes; ++i ) input_port<1>( nodes[i] ).try_put( res[i] );
        g.wait_for_all();

        ASSERT( std::all_of( res.begin(), res.end(), [r]( const opencl_buffer<cl_int> &buf ) {
            return std::all_of( buf.begin(), buf.end(), [r]( cl_int c ) { return c == r; } );
        } ), "Validation has failed" );
    }
    REMARK( "done\n" );
}

void DiamondDependencyTest() {
    REMARK( "DiamondDependencyTest: " );

    const int N = 1 * 1024 * 1024;
    opencl_buffer<cl_short> b( N );
    opencl_buffer<cl_int> b1( N ), b2( N );

    graph g;
    device_selector d;
    opencl_program<> p( PathToFile("test_opencl_node.cl") );
    opencl_node <tuple<opencl_buffer<cl_short>, cl_short>> k0( g, p.get_kernel( "DiamondDependencyTestFill" ), d );
    k0.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );
    opencl_node <tuple<opencl_buffer<cl_short>, opencl_buffer<cl_int>>> k1( g, p.get_kernel( "DiamondDependencyTestSquare" ) );
    k1.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );
    opencl_node <tuple<opencl_buffer<cl_short>, opencl_buffer<cl_int>>> k2( g, p.get_kernel( "DiamondDependencyTestCube" ) );
    k2.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );
    opencl_node <tuple<opencl_buffer<cl_short>, opencl_buffer<cl_int>, opencl_buffer<cl_int>>> k3( g, p.get_kernel( "DiamondDependencyTestDivision" ) );
    k3.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );

    make_edge( output_port<0>( k0 ), input_port<0>( k1 ) );
    make_edge( output_port<0>( k0 ), input_port<0>( k2 ) );
    make_edge( output_port<0>( k0 ), input_port<0>( k3 ) );

    make_edge( output_port<1>( k1 ), input_port<1>( k3 ) );
    make_edge( output_port<1>( k2 ), input_port<2>( k3 ) );

    split_node< tuple<opencl_buffer<cl_short>, cl_short, opencl_buffer<cl_int>, opencl_buffer<cl_int>> > s( g );

    make_edge( output_port<0>( s ), input_port<0>( k0 ) );
    make_edge( output_port<1>( s ), input_port<1>( k0 ) );
    make_edge( output_port<2>( s ), input_port<1>( k1 ) );
    make_edge( output_port<3>( s ), input_port<1>( k2 ) );

    for ( cl_short i = 1; i < 10; ++i ) {
        s.try_put( std::tie( b, i, b1, b2 ) );
        g.wait_for_all();
        ASSERT( std::all_of( b.begin(), b.end(), [i]( cl_short c ) {return c == i*i; } ), "Validation has failed" );
    }
    REMARK( "done\n" );
}

void LoopTest() {
    REMARK( "LoopTest: " );

    const int N = 1 * 1024;
    opencl_buffer<cl_long> b1( N ), b2( N );

    std::fill( b1.begin(), b1.end(), 0 );
    std::fill( b2.begin(), b2.end(), 1 );

    graph g;
    opencl_node <tuple<opencl_buffer<cl_long>, opencl_buffer<cl_long>>> k( g,
        opencl_program<>( PathToFile("test_opencl_node.cl") ).get_kernel( "LoopTestIter" ) );
    k.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );

    make_edge( output_port<1>( k ), input_port<1>( k ) );

    const cl_long numIters = 1000;
    cl_long iter = 0;
    typedef multifunction_node < opencl_buffer<cl_long>, tuple < opencl_buffer<cl_long>, opencl_buffer<cl_long> > > multinode;
    multinode mf( g, serial, [&iter, numIters]( const opencl_buffer<cl_long> &b, multinode::output_ports_type& op ) {
        if ( ++iter < numIters ) get<1>( op ).try_put( b );
        else get<0>( op ).try_put( b );
    } );

    make_edge( output_port<1>( mf ), input_port<0>( k ) );
    make_edge( output_port<0>( k ), mf );

    function_node<opencl_buffer<cl_long>> f( g, unlimited, [numIters]( const opencl_buffer<cl_long> &b ) {
        ASSERT( std::all_of( b.begin(), b.end(), [numIters]( cl_long c ) { return c == numIters*(numIters + 1) / 2; } ), "Validation has failed" );
    } );

    make_edge( output_port<0>( mf ), f );

    split_node< tuple<opencl_buffer<cl_long>, opencl_buffer<cl_long> > > s( g );

    make_edge( output_port<0>( s ), input_port<0>( k ) );
    make_edge( output_port<1>( s ), input_port<1>( k ) );

    s.try_put( std::tie( b1, b2 ) );
    g.wait_for_all();
    REMARK( "done\n" );
}

#include "harness_barrier.h"

template <typename Factory>
struct ConcurrencyTestBodyData {
    typedef opencl_node< tuple<opencl_buffer<cl_char, Factory>, opencl_subbuffer<cl_short, Factory>>, queueing, Factory > NodeType;
    typedef std::vector< NodeType* > VectorType;

    Harness::SpinBarrier barrier;
    VectorType nodes;
    function_node< opencl_subbuffer<cl_short, Factory> > validationNode;
    tbb::atomic<int> numChecks;

    ConcurrencyTestBodyData( graph &g, int numThreads ) : barrier( numThreads ), nodes(numThreads),
        validationNode( g, unlimited, [numThreads, this]( const opencl_subbuffer<cl_short, Factory> &b ) {
            ASSERT( std::all_of( b.begin(), b.end(), [numThreads]( cl_short c ) { return c == numThreads; } ), "Validation has failed" );
            --numChecks;
        } )
    {
        numChecks = 100;
        // The test creates subbers in pairs so numChecks should be even.
        ASSERT( numChecks % 2 == 0, NULL );
    }

    ~ConcurrencyTestBodyData() {
        ASSERT( numChecks == 0, NULL );
        for ( NodeType *n : nodes ) delete n;
    }
};

template <typename Factory>
class ConcurrencyTestBody : NoAssign {
    graph &g;
    std::shared_ptr<ConcurrencyTestBodyData<Factory>> data;
    Factory &f;
    const std::vector<opencl_device> &filteredDevices;

    class RoundRobinDeviceSelector : NoAssign {
    public:
        RoundRobinDeviceSelector( size_t cnt_, int num_checks_, const std::vector<opencl_device> &filteredDevices_ )
            : cnt( cnt_ ), num_checks( num_checks_ ), filteredDevices( filteredDevices_ ) {
        }
        RoundRobinDeviceSelector( const RoundRobinDeviceSelector &src )
            : cnt( src.cnt ), num_checks( src.num_checks ), filteredDevices( src.filteredDevices ) {
            ASSERT( src.num_checks, "The source has already been copied" );
            src.num_checks = 0;
        }
        ~RoundRobinDeviceSelector() {
            ASSERT( !num_checks, "Device Selector has not been called required number of times" );
        }
        opencl_device operator()( Factory &f ) {
            const opencl_device_list& devices = f.devices();
            ASSERT( filteredDevices.size() == devices.size(), "Incorrect list of devices" );
            std::vector<opencl_device>::const_iterator it = filteredDevices.cbegin();
            for ( auto d = devices.begin(); d != devices.end(); ++d ) ASSERT( (*d) == *it++, "Incorrect list of devices" );
            --num_checks;
            return *(devices.begin() + cnt++ % devices.size());
        }
    private:
        size_t cnt;
        mutable int num_checks;
        const std::vector<opencl_device> &filteredDevices;
    };

public:
    ConcurrencyTestBody( graph &g_, int numThreads, Factory &f_, const std::vector<opencl_device> &filteredDevices_ )
        : g( g_ )
        , data( std::make_shared<ConcurrencyTestBodyData<Factory>>( g, numThreads ) )
        , f( f_ )
        , filteredDevices( filteredDevices_ ) {
    }
    void operator()( int idx ) const {
        data->barrier.wait();

        const int N = 1 * 1024;
        const int numChecks = data->numChecks;

        typedef typename ConcurrencyTestBodyData<Factory>::NodeType NodeType;
        NodeType *n1 = new NodeType( g,
            opencl_program<Factory>( f, PathToFile( "test_opencl_node.cl" ) ).get_kernel( "ConcurrencyTestIter" ),
            RoundRobinDeviceSelector( idx, numChecks, filteredDevices ), f );
        // n2 is used to test the copy constructor
        NodeType *n2 = new NodeType( *n1 );
        delete n1;
        data->nodes[idx] = n2;
        n2->set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );

        data->barrier.wait();

        for ( size_t i = 0; i < data->nodes.size() - 1; ++i ) {
            make_edge( output_port<0>( *data->nodes[i] ), input_port<0>( *data->nodes[i + 1] ) );
            make_edge( output_port<1>( *data->nodes[i] ), input_port<1>( *data->nodes[i + 1] ) );
        }
        make_edge( output_port<1>( *data->nodes.back() ), data->validationNode );
        for ( size_t i = 0; i < data->nodes.size() - 1; ++i ) {
            remove_edge( output_port<0>( *data->nodes[i] ), input_port<0>( *data->nodes[i + 1] ) );
            if ( i != (size_t)idx )
                remove_edge( output_port<1>( *data->nodes[i] ), input_port<1>( *data->nodes[i + 1] ) );
        }
        if ( (size_t)idx != data->nodes.size() - 1 )
            remove_edge( output_port<1>( *data->nodes.back() ), data->validationNode );

        data->barrier.wait();
        if ( idx == 0 ) {
            // The first node needs two buffers.
           Harness::FastRandom rnd(42);
            cl_uint alignment = 0;
            for (  auto d = filteredDevices.begin(); d != filteredDevices.end(); ++d ) {
                cl_uint deviceAlignment;
                (*d).info( CL_DEVICE_MEM_BASE_ADDR_ALIGN, deviceAlignment );
                alignment = max( alignment, deviceAlignment );
            }
            alignment /= CHAR_BIT;
            cl_uint alignmentMask = ~(alignment-1);
            for ( int i = 0; i < numChecks; i += 2 ) {
                for ( int j = 0; j < 2; ++j ) {
                    opencl_buffer<cl_char, Factory> b1( f, N );
                    std::fill( b1.begin(), b1.end(), cl_char(1) );
                    input_port<0>( *n2 ).try_put( b1 );
                }

                // The subbers are created in pairs from one big buffer
                opencl_buffer<cl_short, Factory> b( f, 4*N );
                size_t id0 = (rnd.get() % N) & alignmentMask;
                opencl_subbuffer<cl_short, Factory> sb1( b, id0, N );
                std::fill( sb1.begin(), sb1.end(), cl_short(0) );
                input_port<1>( *n2 ).try_put( sb1 );

                size_t id1 = (rnd.get() % N) & alignmentMask;
                opencl_subbuffer<cl_short, Factory> sb2 = b.subbuffer( 2*N + id1, N );
                std::fill( sb2.begin(), sb2.end(), cl_short(0) );
                input_port<1>( *n2 ).try_put( sb2 );
            }
        } else {
            // Other nodes need only one buffer each because input_port<1> is connected with
            // output_port<1> of the previous node.
            for ( int i = 0; i < numChecks; ++i ) {
                opencl_buffer<cl_char, Factory> b( f, N );
                std::fill( b.begin(), b.end(), cl_char(1) );
                input_port<0>( *n2 ).try_put( b );
            }
        }

        g.wait_for_all();

        // n2 will be deleted in destructor of ConcurrencyTestBodyData
    }
};

const int concurrencyTestNumRepeats = 5;

template <typename Factory = DefaultFactoryType>
void ConcurrencyTest( const std::vector<opencl_device> &filteredDevices ) {
    const int numThreads = min( tbb::task_scheduler_init::default_num_threads(), 8 );
    for ( int i = 0; i < concurrencyTestNumRepeats; ++i ) {
        tbb::task_group_context ctx( tbb::task_group_context::isolated, tbb::task_group_context::default_traits | tbb::task_group_context::concurrent_wait );
        graph g( ctx );
        opencl_device_list dl;
        Factory f;
        ConcurrencyTestBody<Factory> body( g, numThreads, f, filteredDevices );
        NativeParallelFor( numThreads, body );
    }
}

#include <unordered_map>

enum FilterPolicy {
    MAX_DEVICES,
    ONE_DEVICE
};

template <FilterPolicy Policy>
struct DeviceFilter {
    DeviceFilter() {
        filteredDevices.clear();
    }
    opencl_device_list operator()( opencl_device_list device_list ) {
        ASSERT( filteredDevices.size() == 0, NULL );
        switch ( Policy ) {
        case MAX_DEVICES:
        {
            std::unordered_map<std::string, std::vector<opencl_device>> platforms;
            for (auto d = device_list.begin(); d != device_list.end(); ++d) {
                platforms[(*d).platform_name()].push_back(*d);
            }

            // Select a platform with maximum number of devices.
            filteredDevices = std::max_element( platforms.begin(), platforms.end(),
                []( const std::pair<std::string, std::vector<opencl_device>>& p1, const std::pair<std::string, std::vector<opencl_device>>& p2 ) {
                return p1.second.size() < p2.second.size();
            } )->second;

            if ( !numRuns ) {
                REMARK( "  Chosen devices from the same platform (%s):\n", filteredDevices[0].platform_name().c_str() );
                for ( auto d = filteredDevices.begin(); d != filteredDevices.end(); d++ ) {
                    REMARK( "    %s\n", (*d).name().c_str() );
                }
            }

            if ( filteredDevices.size() < 2 )
                REPORT_ONCE( "Known issue: the system does not have several devices in one platform\n" );
            break;
        }
        case ONE_DEVICE:
        {
            ASSERT( deviceNum < device_list.size(), NULL );
            opencl_device_list::iterator it = device_list.begin();
            std::advance( it, deviceNum );
            filteredDevices.push_back( *it );
            break;
        }
        default:
            ASSERT( false, NULL );
        }
        opencl_device_list dl;
        for ( auto d = filteredDevices.begin(); d != filteredDevices.end(); ++d ) dl.add( *d );

        ++numRuns;

        return dl;
    }
    static opencl_device_list::size_type deviceNum;
    static int numRuns;
    static std::vector<opencl_device> filteredDevices;
};

template <FilterPolicy Policy>
opencl_device_list::size_type DeviceFilter<Policy>::deviceNum;
template <FilterPolicy Policy>
int DeviceFilter<Policy>::numRuns;
template <FilterPolicy Policy>
std::vector<opencl_device> DeviceFilter<Policy>::filteredDevices;

void CustomFactoryTest() {
    REMARK( "CustomFactoryTest:\n" );
    REMARK( "  Multi device test:\n" );
    DeviceFilter<MAX_DEVICES>::numRuns = 0;
    typedef tbb::flow::opencl_factory <DeviceFilter<MAX_DEVICES>> custom_factory;
    ConcurrencyTest<custom_factory>( DeviceFilter<MAX_DEVICES>::filteredDevices );
    ASSERT( DeviceFilter<MAX_DEVICES>::numRuns == concurrencyTestNumRepeats, NULL );

    REMARK( "  One device tests:\n" );
    graph g;
    opencl_device_list all_devices = interface10::opencl_info::available_devices();
    for ( int i = 0; i < (int)all_devices.size(); ++i ) {
        opencl_device_list::const_iterator it = all_devices.begin();
        std::advance( it, i );
        REMARK( "    %s: ", it->name().c_str() );
        DeviceFilter<ONE_DEVICE>::numRuns = 0;
        DeviceFilter<ONE_DEVICE>::deviceNum = i;
        typedef tbb::flow::opencl_factory <DeviceFilter<ONE_DEVICE>> one_device_factory;
        ConcurrencyTest<one_device_factory>( DeviceFilter<ONE_DEVICE>::filteredDevices );
        ASSERT( DeviceFilter<ONE_DEVICE>::numRuns == concurrencyTestNumRepeats, NULL );
        ASSERT( DeviceFilter<ONE_DEVICE>::filteredDevices[0] == *it, NULL );
        REMARK( "done\n" );
    }
    REMARK( "CustomFactoryTest: done\n" );
}

void DefaultConcurrencyTest() {
    REMARK( "DefaultConcurrencyTest: " );
    // By default, the first device is selected.
    ConcurrencyTest( { *interface10::opencl_info::available_devices().begin() } );
    REMARK( "done\n" );
}


void SpirKernelTest() {
    REMARK( "SpirKernelTest:\n" );

    const opencl_device_list devices = interface10::opencl_info::available_devices();

    for( auto d = devices.begin(); d != devices.end(); d++ ) {
        if( !(*d).extension_available( "cl_khr_spir" ) ) {
            REMARK( "  Extension 'cl_khr_spir' is not available on the device '%s'\n", (*d).name().c_str() );
            continue;
        }

        graph g;
        DefaultFactoryType factory;

        bool init = factory.init( { *d } );
        ASSERT( init, "It should be the first initialization" );

        std::string path_to_file = PathToFile(std::string("test_opencl_kernel_") +
                                              std::to_string((*d).address_bits()) + std::string(".spir") );
        REMARK("  Using SPIR file '%s' on device '%s'\n", path_to_file.c_str(), (*d).name().c_str());
        const int N = 1 * 1024 * 1024;
        opencl_buffer<float, DefaultFactoryType> b1( factory, N ), b2( factory, N );
        std::vector<float> v1( N ), v2( N );

        auto i1 = b1.access<write_only>();
        auto i2 = b2.access<write_only>();

        for ( int i = 0; i < N; ++i ) {
            i1[i] = v1[i] = float( i );
            i2[i] = v2[i] = float( 2 * i );
        }

        typedef opencl_node< tuple<opencl_buffer<float, DefaultFactoryType>, opencl_buffer<float, DefaultFactoryType> >, queueing, DefaultFactoryType > OpenCLNodeType;

        OpenCLNodeType k1( g, opencl_program<DefaultFactoryType>( factory, opencl_program_type::SPIR, path_to_file ).get_kernel( "custom_summer" ), factory );
        k1.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );

        input_port<0>(k1).try_put( b1 );
        input_port<1>(k1).try_put( b2 );

        g.wait_for_all();

        // validation
        for ( int i = 0; i < N; ++i ) {
            v2[i] += v1[i];
        }

        ASSERT( memcmp( &b2[0], &v2[0], N*sizeof( float ) ) == 0, "Validation has failed" );
    }
    REMARK( "done\n" );
}

void PrecompiledKernelTest() {
    REMARK( "PrecompiledKernelTest:\n" );

    graph g;
    DefaultFactoryType factory;

    const opencl_device_list devices = interface10::opencl_info::available_devices();
    opencl_device_list::const_iterator it = std::find_if(
        devices.cbegin(), devices.cend(),
        []( const opencl_device &d ) {
            std::string vendor_name = d.vendor();
            return std::string::npos != vendor_name.find( "Intel" ) && CL_DEVICE_TYPE_GPU == d.type();
        } );

    if ( it == devices.cend() ) {
        REPORT( "Known issue: there is no device in the system that supports the precompiled GPU kernel.\n" );
        return;
    }
    bool init = factory.init( { *it } );
    ASSERT( init, "It should be the first initialization" );
    REMARK( "  Device name '%s', %s:", it->name().c_str(), it->version().c_str() );

    const int N = 1 * 1024 * 1024;
    opencl_buffer<float, DefaultFactoryType> b1( factory, N ), b2( factory, N );
    std::vector<float> v1( N ), v2( N );

    auto i1 = b1.access<write_only>();
    auto i2 = b2.access<write_only>();

    for ( int i = 0; i < N; ++i ) {
        i1[i] = v1[i] = float( i );
        i2[i] = v2[i] = float( 2 * i );
    }

    std::string path_to_file = PathToFile(std::string("test_opencl_precompiled_kernel_gpu_") + std::to_string((*it).address_bits()) + std::string(".ir"));

    opencl_program<DefaultFactoryType> p( factory, opencl_program_type::PRECOMPILED, path_to_file);
    opencl_node < tuple<opencl_buffer<float, DefaultFactoryType>, opencl_buffer<float, DefaultFactoryType> >, queueing, DefaultFactoryType > k1(g, p.get_kernel("custom_subtractor"), factory);
    k1.set_range({ BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) });

    input_port<0>(k1).try_put( b1 );
    input_port<1>(k1).try_put( b2 );

    g.wait_for_all();

    // validation
    for ( int i = 0; i < N; ++i ) {
        v2[i] -= v1[i];
    }

    ASSERT( memcmp( &b2[0], &v2[0], N*sizeof( float ) ) == 0, "Validation has failed" );
    REMARK( " done\n" );
}

/*
    /--functional_node-\   /-functional_node-\                       /--functional_node-\
    |                  |   |                 |   /--opencl_node--\   |                  |
    O Buffer generator O---O  Buffer filler  O---O               O---O Result validator O
    |                  |   |                 |   |               |   |                  |
    \------------------/   \-----------------/   |               |   \------------------/
    |   Multiplier  |
    /--functional_node-\   /-functional_node-\   |               |
    |                  |   |                 |   |               |
    O Buffer generator O---O  Buffer filler  O---O               O
    |                  |   |                 |   \---------------/
    \------------------/   \-----------------/
    */

template <typename Key>
struct BufferWithKey : public opencl_buffer<int> {
    typedef typename std::decay<Key>::type KeyType;
    KeyType my_key;
    int my_idx;

    // TODO: investigate why defaul ctor is required
    BufferWithKey() {}
    BufferWithKey( size_t N, int idx ) : opencl_buffer<int>( N ), my_idx( idx ) {}
    const KeyType& key() const { return my_key; }
};

template <typename Key>
Key KeyGenerator( int i );

template <>
int KeyGenerator<int>( int i ) { return i; }

template <>
std::string KeyGenerator<std::string>( int i ) { return std::to_string( i ); }

template <typename Key>
BufferWithKey<Key> GenerateRandomBuffer( BufferWithKey<Key> b ) {
    b.my_key = KeyGenerator<typename std::decay<Key>::type>( b.my_idx );
    Harness::FastRandom r( b.my_idx );
    std::generate( b.begin(), b.end(), [&r]() { return r.get(); } );
    return b;
}

template <typename Key, typename JP>
bool KeyMatchingTest() {
    const int N = 1000;
    const int numMessages = 100;

    graph g;
    broadcast_node<int> b( g );

    // Use opencl_async_msg's to have non-blocking map to host
    function_node<int, opencl_async_msg<BufferWithKey<Key>>>
        bufGenerator1( g, unlimited, [N]( int i ) { return opencl_async_msg<BufferWithKey<Key>>( BufferWithKey<Key >(N, i) ); } ),
        bufGenerator2 = bufGenerator1;

    function_node<BufferWithKey<Key>, BufferWithKey<Key>>
        bufFiller1( g, unlimited, []( const BufferWithKey<Key> &b ) { return GenerateRandomBuffer<Key>( b ); } ),
        bufFiller2 = bufFiller1;

    opencl_node< tuple< BufferWithKey<Key>, BufferWithKey<Key> >, JP > k( g,
        opencl_program<>( PathToFile( "test_opencl_node.cl" ) ).get_kernel( "Mul" ) );
    k.set_range( { BROKEN_INITIALIZER_LIST_DEDUCTION({ N }) } );

    bool success = true;
    function_node<BufferWithKey<Key>> checker( g, unlimited, [&success, N]( BufferWithKey<Key> b ) {
        Harness::FastRandom r( b.my_idx );
        std::for_each( b.begin(), b.end(), [&success, &r]( int bv ) {
            const int rv = r.get();
            if ( bv != rv*rv ) {
                success = false;
                return;
            }
        } );
    } );

    make_edge( bufGenerator1, bufFiller1 );
    make_edge( bufGenerator2, bufFiller2 );
    make_edge( bufFiller1, input_port<0>( k ) );
    make_edge( bufFiller2, input_port<1>( k ) );
    make_edge( output_port<0>( k ), checker );

    for ( int i = 0; i < numMessages; ++i ) {
        bufGenerator1.try_put( i );
        bufGenerator2.try_put( numMessages - i - 1 );
    }

    g.wait_for_all();

    return success;
}

void KeyMatchingTest() {
    REMARK( "KeyMatchingTest:\n" );
    REMARK( "  Queueing negative test: " );
    bool res = !KeyMatchingTest<int, queueing>(); // The test should fail with the queueing policy, so the negative result is expected.
    ASSERT( res, "Queueing negative test has failed" );
    REMARK( "done\n  key_matching<int> test: " );
    res = KeyMatchingTest<int, key_matching<int>>();
    ASSERT( res, "key_matching<int> test has failed" );
    REMARK( "done\n  key_matching<string&> test: " );
    res = KeyMatchingTest<std::string&, key_matching<std::string&>>();
    ASSERT( res, "key_matching<string&> test has failed" );
    REMARK( "done\n" );
    REMARK( "KeyMatchingTest: done\n" );
}

int TestMain() {
    TestArgumentPassing();

    SimpleDependencyTest();
    BroadcastTest();
    DiamondDependencyTest();
    LoopTest();

    DefaultConcurrencyTest();
    CustomFactoryTest();

    SpirKernelTest();
#if !__APPLE__
    // Consider better support for precompiled programs on Apple
    PrecompiledKernelTest();
#endif

    KeyMatchingTest();

    return Harness::Done;
}
#else
#define HARNESS_SKIP_TEST 1
#include "harness.h"
#endif
