//DIAGNOSTIC_TEST(dxc):SIMPLE(filecheck=CHECK):-pass-through dxc -target dxil -entry computeMain -stage compute -profile sm_6_1

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
    // Error should be here... as gOutputBuffer is not defined...
	gOutputBuffer[tid] = dispatchThreadID.x * 0.5f;
  // CHECK:      : tests/cross-compile/dxc-error.hlsl([[#@LINE-1]]): error :  use of undeclared identifier 'gOutputBuffer'
  // CHECK-NEXT: : note :         gOutputBuffer[tid] = dispatchThreadID.x * 0.5f;
  // CHECK-NEXT: : note :         ^
}
