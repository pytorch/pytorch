// image-load.slang

// This test confirms that `.Load()` on a `RWTexure*`
// gets properly converted to a call to `imageLoad`
// and not just `texelFetch` as it would for a `Texture*`.

//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage compute -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage compute

struct Params
{
	RWTexture2DArray<float> tex;
}

ParameterBlock<Params> gParams;

void main(uint3 tid : SV_DispatchThreadID)
{
    // CHECK: OpImageRead
    // CHECK: OpImageWrite
    float f = gParams.tex.Load(int3(int2(tid.xy), int(tid.z)));
    gParams.tex[tid] = f + 1.0;
}
