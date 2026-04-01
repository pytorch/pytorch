// parameter-block.slang

struct CB
{
    uint4 value;
}

struct MaterialSystem
{
    CB cb;
    StructuredBuffer<uint4> data;
}

struct Scene
{
    CB sceneCb;
    StructuredBuffer<uint4> data;
    ParameterBlock<MaterialSystem> material;
}

cbuffer PerView
{
    uint4 value;
}

ParameterBlock<Scene> scene;

RWStructuredBuffer<uint4> resultBuffer;

// Main entry-point. Applies the transformation encoded by `transformer`
// to all elements in `buffer`.
[shader("compute")]
[numthreads(4,1,1)]
void computeMain(uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    resultBuffer[sv_dispatchThreadID.x] = value.x + scene.sceneCb.value.x + scene.data[0].x + scene.material.cb.value.x + scene.material.data[0].x;
}
