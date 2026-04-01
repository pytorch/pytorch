//TEST:SIMPLE(filecheck=CHECK): -target spirv

//CHECK: OpEntryPoint

interface IBsdf {};
struct Foo : IBsdf {}
//TEST_INPUT:type_conformance Foo:IBsdf = 0
struct Mesh {
    float4 *vertices;
    IBsdf *bsdf;
}
[[vk::push_constant]] Mesh* mesh;
RWStructuredBuffer<float4> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = mesh.vertices[0];
}