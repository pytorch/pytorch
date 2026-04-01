//TEST:SIMPLE(filecheck=CHECK): -target spirv -fvk-use-entrypoint-name

// CHECK: OpEntryPoint

RWStructuredBuffer<int> output;

[shader("compute")]
[numthreads(1,1,1)]
void innerMain(int id : SV_DispatchThreadID)
{
    output[id] = id;
}

[shader("compute")]
[numthreads(1,1,1)]
void outerMain(int id : SV_DispatchThreadID)
{
    innerMain(id);
}