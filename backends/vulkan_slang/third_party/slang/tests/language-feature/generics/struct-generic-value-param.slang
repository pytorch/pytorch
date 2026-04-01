// struct-generic-value-param.slang

// This test reproduces a few bugs related to declarations
// not being emitted IR and specialized correctly.
//
// First, it tests that a method in a generic `struct`
// gets properly emitted to the IR of its own module.
//
// Second, it tests that witness tables for an empty
// `interface` can be emitted and used to specialize
// code with generic type parameters constrained to
// those interfaces.

// This file is attempting to stress-test use of a `struct`
// type with a generic value parameter. In particular, it
// it can reproduce a bug that was encountered by a user
// when trying out the feature.

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

import struct_generic_value_param_import;

Data<N> makeData<let N : int>( int val )
{
    Data<N> result = { val };
    return result;
}

void doThings<D : IData>(D data, inout int v)
{
    v++;
}

int test(int val)
{
    var data = makeData<4>(val);

    // Note: with the original bug, this call emitted
    // as `/* unhandled */(data)` which meant the call
    // acted as a no-op.
    //
    data.doStuff();

    // Note: with the original bug, this call emitted
    // as `/* uhandled */(data, val)` which is also
    // a no-op (that happens to use `operator,`).
    //
    doThings(data, val);

    return data.state + val*16;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
