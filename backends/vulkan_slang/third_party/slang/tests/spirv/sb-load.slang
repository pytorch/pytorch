//TEST:SIMPLE(filecheck=CHECK): -target spirv

#define FILL_PATTERN_DIMENSIONS_X 16
#define FILL_PATTERN_DIMENSIONS_Y 16

struct FillPatternBuffer
{
    float4 px[FILL_PATTERN_DIMENSIONS_Y][FILL_PATTERN_DIMENSIONS_X];
};

StructuredBuffer<FillPatternBuffer> dp;
RWStructuredBuffer<float4> outputBuffer;

// CHECK-NOT: OpCompositeConstruct

[numthreads(4, 4, 1)]
void main(uint3 GTid : SV_GroupThreadID, 
          uint GI    : SV_GroupIndex)
{
    const uint ii = GTid.x;
    const uint jj = GTid.y;
    const float4 pmv = dp[0].px[ii][jj];
    outputBuffer[GI] = pmv;
}