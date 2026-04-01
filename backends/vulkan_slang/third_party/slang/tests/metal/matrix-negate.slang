//TEST:SIMPLE(filecheck=CHECK): -target metal

// CHECK: (matrix<float,int(4),int(4)>{{.*}}(0) -

RWStructuredBuffer<float4> output;

[numthreads(1,1,1)]
void computeMain(uniform float4x4 m)
{
    output[0] = (-m)[0];
}