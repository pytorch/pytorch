//TEST:SIMPLE(filecheck=METAL): -stage compute -entry computeMain -target metal -DEMIT_SOURCE
//TEST:SIMPLE(filecheck=METALLIB): -stage compute -entry computeMain -target metallib
// for some reason, metal textures dont have an overload for less-than-four component
// writes, they need to be converted to 4-components in a legalize step, as the other components
// get discarded
struct TextureWrite
{
    //TEST_INPUT: RWTexture2D(size=4, content = zero):name t2D_f32
    RWTexture2D<float> tex1;
    //TEST_INPUT: RWTexture2D(size=4, content = zero):name t2D_f32v2
    RWTexture2D<float2> tex2;
    //TEST_INPUT: RWTexture2D(size=4, content = zero):name t2D_f32v3
    RWTexture2D<float3> tex3;
    //TEST_INPUT: RWTexture2D(size=4, content = zero):name t2D_f32v4
    RWTexture2D<float4> tex4;

    //TEST_INPUT: RWTexture2DArray(size=4, content = zero):name t2D_f32
    RWTexture2DArray<float> tex1Array;
    //TEST_INPUT: RWTexture2DArray(size=4, content = zero):name t2D_f32v2
    RWTexture2DArray<float2> tex2Array;
    //TEST_INPUT: RWTexture2DArray(size=4, content = zero):name t2D_f32v3
    RWTexture2DArray<float3> tex3Array;
    //TEST_INPUT: RWTexture2DArray(size=4, content = zero):name t2D_f32v4
    RWTexture2DArray<float4> tex4Array;
}
ParameterBlock<TextureWrite> pWrites;

[numthreads(1, 1, 1)]
void computeMain()
{
    // TODO: check for the type of first parameter to be a 4-component vector
    // METALLIB: call {{.*}}.write_texture_2d.v4f32(
    // METAL: .write(
    pWrites.tex1[uint2(1, 1)] = 1;
    // METALLIB: call {{.*}}.write_texture_2d.v4f32(
    // METAL: .write(
    pWrites.tex2[uint2(2, 2)] = float2(1, 2);
    // METALLIB: call {{.*}}.write_texture_2d.v4f32(
    // METAL: .write(
    pWrites.tex3[uint2(3, 3)] = float3(1, 2, 3);
    // METALLIB: call {{.*}}.write_texture_2d.v4f32(
    // METAL: .write(
    pWrites.tex4[uint2(4, 4)] = float4(1, 2, 3, 4);

    // METALLIB: call {{.*}}.write_texture_2d_array.v4f32(
    // METAL: .write(
    pWrites.tex1Array[uint3(1, 1, 1)] = 1;
    // METALLIB: call {{.*}}.write_texture_2d_array.v4f32(
    // METAL: .write(
    pWrites.tex2Array[uint3(2, 2, 2)] = float2(1, 2);
    // METALLIB: call {{.*}}.write_texture_2d_array.v4f32(
    // METAL: .write(
    pWrites.tex3Array[uint3(3, 3, 3)] = float3(1, 2, 3);
    // METALLIB: call {{.*}}.write_texture_2d_array.v4f32(
    // METAL: .write(
    pWrites.tex4Array[uint3(4, 4, 4)] = float4(1, 2, 3, 4);
}