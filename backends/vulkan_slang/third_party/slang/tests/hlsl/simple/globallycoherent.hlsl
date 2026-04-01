//TEST:COMPARE_HLSL:-profile cs_5_0

// Check output for `globallycoherent`

#ifndef __SLANG__
#define gBuffer gBuffer_0
#define SV_DispatchThreadID SV_DISPATCHTHREADID
#endif

globallycoherent
RWStructuredBuffer<uint> gBuffer : register(u0);

[numthreads(16,1,1)]
void main(
	uint tid : SV_DispatchThreadID)
{
	uint index = tid;

    gBuffer[index] = gBuffer[index + 1];
}
