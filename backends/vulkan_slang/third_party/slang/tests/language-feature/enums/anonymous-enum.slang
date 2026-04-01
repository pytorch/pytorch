//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpEntryPoint

enum
{
    Red,Green,Blue
}

RWStructuredBuffer<int> buffer;
[numthreads(1, 1, 1)]
void main(int id: SV_DispatchThreadID)
{
    if (id == Red)
    {
        buffer[0] = Green;
    }
    buffer[1] = Blue;
}
