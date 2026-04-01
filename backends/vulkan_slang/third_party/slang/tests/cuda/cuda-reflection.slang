// cuda-reflection.slang

//TEST:REFLECTION:-stage compute -entry main -target cuda -no-codegen

struct PadLadenStruct
{
    double a;
    uint8_t b;
};

// This is to check if the last half can be inserted 'inside' the spare padding of a. It should not be
struct StructWithArray
{
    PadLadenStruct a[1];
    uint8_t c;
    
    matrix<half, 3, 3> d;
    uint8_t e;
};

ConstantBuffer<StructWithArray> cb;
RWStructuredBuffer<StructWithArray> sb;

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
}