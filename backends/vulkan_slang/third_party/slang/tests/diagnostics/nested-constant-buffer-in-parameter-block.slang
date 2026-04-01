//TEST:SIMPLE(filecheck=CHECK): -target metal

struct T
{
    RWStructuredBuffer<int> buf;
}
    
struct U
{
    ConstantBuffer<T> t;
}

struct S
{
    ConstantBuffer<RWStructuredBuffer<int>> cb;
}

// CHECK-DAG: ([[# @LINE+1]]): error 56101:
ParameterBlock<S> s1;

// CHECK-DAG: ([[# @LINE+1]]): error 56101:
ParameterBlock<U> s2;

// CHECK-DAG: ([[# @LINE+1]]): error 56101:
ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>> s3;

RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void kernelMain()
{
    outputBuffer[0] = s1.cb[0] + s2.t.buf[0] + s3[0];
}
