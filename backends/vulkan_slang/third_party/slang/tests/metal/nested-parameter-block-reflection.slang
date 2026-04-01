//TEST:REFLECTION(filecheck=CHECK): -target metal

// CHECK:"name": "tex",
// CHECK:"binding": {"kind": "metalArgumentBufferElement", "index": 1}

struct Data { int3 content; }
struct Params 
{
    ParameterBlock<Data> pdata;
    Texture2D tex;
}
ParameterBlock<Params> gParams;
RWStructuredBuffer<float4> output;
[numthreads(1,1,1)]
void computeMain()
{
    output[0] = gParams.tex.Load(gParams.pdata.content);
}