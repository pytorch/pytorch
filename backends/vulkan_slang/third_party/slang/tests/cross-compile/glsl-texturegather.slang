//TEST:SIMPLE(filecheck=CHECK): -stage fragment -entry main -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK): -stage fragment -entry main -target spirv

// CHECK: OpImageDrefGather
// CHECK: OpImageDrefGather
// CHECK: OpImageDrefGather
// CHECK: OpImageDrefGather

Texture2D t;
SamplerState s;
SamplerComparisonState sc;
float4 main()
{
    const float2 loc = float2(1,2);
    const float cmp = 3;
    const int2 off = int2(4,5);
    return t.GatherCmp(sc, loc, cmp)
         + t.GatherCmp(sc, loc, cmp, off)
         + t.GatherCmp(sc, loc, cmp, off, off, off, off)
         + t.GatherCmpRed(sc, loc, cmp, int2(6,6), int2(7,7), int2(8,8), int2(9,9));
}
