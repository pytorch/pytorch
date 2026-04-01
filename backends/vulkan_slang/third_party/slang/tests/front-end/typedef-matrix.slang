//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry main -stage compute -matrix-layout-row-major

// CHECK: ColMajor

typedef column_major float3x4 Mat;

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void main(uniform Mat m)
{
    output[0] = m[0][0];
}