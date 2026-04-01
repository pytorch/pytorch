//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly -O0
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-via-glsl -entry main -stage compute -O0

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
[shader("compute")]
void main()
{
    output[0] = 0;
    // CHECK: OpControlBarrier %uint_2 %uint_2 %uint_264
    GroupMemoryBarrierWithGroupSync();
    output[1] = 0;
}