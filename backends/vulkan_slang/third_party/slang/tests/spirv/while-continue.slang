//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -emit-spirv-directly

// Test that we can transform a continue in a while loop to valid spirv.

uniform int loopCount;

// CHECK: OpEntryPoint

bool condition1()
{
    AllMemoryBarrier();
    return output[2] < 1;
}
bool condition2()
{
    AllMemoryBarrier();
    return output[3] < 1;
}

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void main()
{
    float weight = 0.0;
    do 
    {
        if (condition2()) 
        { 
            continue;
        }
        AllMemoryBarrier();
    } while (condition1());

    while(condition1())
    {
        if (condition2())
        {
            continue;
        }
        AllMemoryBarrier();
    }

    output[0] = weight;
}
