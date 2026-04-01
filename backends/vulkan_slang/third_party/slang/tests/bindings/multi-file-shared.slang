// multi-file-shared.slang
//TEST_IGNORE_FILE:

#include "multi-file-defines.h"

PUBLIC float4 use(float  val) { return val; };
PUBLIC float4 use(float2 val) { return float4(val,0.0,0.0); };
PUBLIC float4 use(float3 val) { return float4(val,0.0); };
PUBLIC float4 use(float4 val) { return val; };
PUBLIC float4 use(Texture2D t, SamplerState s) { return t.SampleLevel(s, 0.0, 0.0); }

PUBLIC Texture2D sharedT R(: register(t2));
PUBLIC SamplerState sharedS R(: register(s2));

PUBLIC BEGIN_CBUFFER(sharedC)
{
    PUBLIC float3 sharedCA;
    PUBLIC float sharedCB;
    PUBLIC float3 sharedCC;
    PUBLIC float2 sharedCD;
}
END_CBUFFER(sharedC, register(b2))

PUBLIC Texture2D sharedTV R(: register(t3));
PUBLIC Texture2D sharedTF R(: register(t4));
