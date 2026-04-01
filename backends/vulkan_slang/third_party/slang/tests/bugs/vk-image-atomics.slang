//TEST:SIMPLE(filecheck=CHECK): -profile ps_5_0 -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -profile ps_5_0 -entry main -target spirv-assembly -emit-spirv-via-glsl

// Ensure that we can lower to `imageAtomicAdd` correctly.

// CHECK: OpImageTexelPointer
// CHECK: OpAtomicIAdd

RWTexture2D<uint> t;

float4 main() : SV_Target
{
	uint u;
	InterlockedAdd(t[uint2(0)], 1, u);
	return u;
}
