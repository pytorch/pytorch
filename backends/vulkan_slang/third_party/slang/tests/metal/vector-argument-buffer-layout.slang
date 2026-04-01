//TEST:REFLECTION(filecheck=CHECK): -target metal
//CHECK: "binding": {"kind": "metalArgumentBufferElement", "index": 0}
//CHECK: "binding": {"kind": "metalArgumentBufferElement", "index": 1}

struct Params
{
    float3 dir;
    float4 color;
}

RWStructuredBuffer<float> o;
ParameterBlock<Params> gParams;

[numthreads(1,1,1)]
void computeMain()
{
    o[0] = gParams.dir.x;
}