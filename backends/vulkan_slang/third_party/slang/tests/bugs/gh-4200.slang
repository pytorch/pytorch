//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -entry computeMain -stage compute

//CHECK_SPV: OpEntryPoint
//CHECK_HLSL: computeMain

Texture2DMS<uint4> Src : register(t1);
RWBuffer<uint4>    Dst : register(u2);

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
// CHECK_SPV: Sample %int_0
// CHECK_HLSL: [{{.*}}1{{.*}}]
    Dst[0] = Src[int2(1)];
}