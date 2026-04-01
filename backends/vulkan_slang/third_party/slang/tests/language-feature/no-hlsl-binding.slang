//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry main -profile cs_6_0 -no-mangle -no-hlsl-binding

// Test that -no-hlsl-binding option disables automatically assigned bindings in the hlsl output.

// CHECK-DAG: cbuffer {{.*}} : register(b3)
cbuffer MyCB : register(b3)
{
    int member;
}

// CHECK-DAG: RWStructuredBuffer<float > outputBuffer : register(u2);
RWStructuredBuffer<float> outputBuffer : register(u2);

// CHECK-DAG: RWStructuredBuffer<int > inputBuffer;
RWStructuredBuffer<int> inputBuffer;

[numthreads(1,1,1)]
void main()
{
    outputBuffer[0] = inputBuffer[0] + member;
}
