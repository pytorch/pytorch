//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry main -profile cs_6_0 -no-mangle

// Test that -no-mangle option disables name mangling on global parameters and struct fields.

cbuffer BufferName
{
    int cbufferParam;
}

struct MyStruct
{
    int v;
}

RWStructuredBuffer<MyStruct> outputBuffer;

// CHECK-DAG: SLANG_ParameterGroup_BufferName{{.*}} BufferName;

// CHECK-DAG: RWStructuredBuffer<{{.*}}> outputBuffer : 

[numthreads(1,1,1)]
void main()
{
    // CHECK-DAG: = {{.*}}cbufferParam; 

    outputBuffer[0].v = cbufferParam;
}