//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj -output-using-type

// This test confirms that we can provide a subset of the required generic
// arguments to a generic function, and have the rest be inferred from the
// types of the value arguments.

interface IConvertible
{
    __init( int value );

    property value : int { get; }
}

ToType convert< ToType : IConvertible, FromType : IConvertible >( FromType fromVal )
{
    return ToType( fromVal.value*0x10 );
}

struct A : IConvertible
{
    int value;
    __init(int v) { this.value = v; }
}

struct B : IConvertible
{
    int value;
    __init(int v) { this.value = v; }
}


int test( int value )
{
    A a = A(value);
    B b = convert<B>(a);

    return a.value + b.value;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    outputBuffer[tid] = test(tid);
}
