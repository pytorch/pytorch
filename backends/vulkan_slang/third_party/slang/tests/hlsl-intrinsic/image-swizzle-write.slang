//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target spirv

layout(rgba8)
RWTexture2D<float4> texture;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    half h = 1.0h;
    // CHECK: OpImageRead
    // CEHCK: OpCompositeInsert
    // CHECK: OpImageWrite
    texture[dispatchThreadID.xy].a = h;
}
