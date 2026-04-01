//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -stage compute -profile glsl_450 -target spirv -entry computeMain -warnings-disable 39001

//CHECK-NOT: warning
//CHECK-NOT: note
RWStructuredBuffer<float> outputBuffer;
[[vk::binding(1,0)]] Texture2D g_bindlessTexture2D[];
[[vk::binding(1,0)]] Texture3D g_bindlessTexture3D[];

struct GenStruct<T>
{
	T x;
};

T test<T>(T val)
{
	return val;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	float inVal = float(tid);

	float outVal = test<float>(inVal);

	outputBuffer[tid] = outVal;
}