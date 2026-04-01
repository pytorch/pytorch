//TEST_IGNORE_FILE:

#include "multi-file-defines.h"

#ifdef __SLANG__
import multi_file_shared;
#else
#include "multi-file-shared.slang"
#endif

Texture2D fragmentT R(: register(t1));
SamplerState fragmentS R(: register(s1));

BEGIN_CBUFFER(fragmentC)
{
	float3 fragmentCA;
	float  fragmentCB;
	float3 fragmentCC;
	float2 fragmentCD;
}
END_CBUFFER(fragmentC, register(b1))

float4 mainFS() : SV_TARGET
{
	// Go ahead and use everything here, just to make sure things got placed correctly
	return use(sharedT, sharedS)
		+  use(CBUFFER_REF(sharedC,sharedCD))
		+  use(fragmentT, fragmentS)
		+  use(CBUFFER_REF(fragmentC, fragmentCD))
		+  use(sharedTF, sharedS)
		;
}