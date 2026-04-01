//TEST:SIMPLE(filecheck=SPIRV): -stage compute -target spirv -emit-spirv-directly -entry main
//TEST:SIMPLE(filecheck=SPIRV): -stage compute -target spirv -entry main
// Confirm that all HLSL barrier operations can be used when cross-compiling.

[numthreads(1,1,1)]
void main()
{
    // SPIRV: OpMemoryBarrier %uint_1 %uint_2376
    // SPIRV: OpControlBarrier %uint_2 %uint_1 %uint_2376
    // SPIRV: OpMemoryBarrier %uint_1 %uint_2120
    // SPIRV: OpControlBarrier %uint_2 %uint_1 %uint_2120
    // SPIRV: OpMemoryBarrier %uint_2 %uint_264
    // SPIRV: OpControlBarrier %uint_2 %uint_2 %uint_264
	AllMemoryBarrier();
	AllMemoryBarrierWithGroupSync();
	DeviceMemoryBarrier();
	DeviceMemoryBarrierWithGroupSync();
	GroupMemoryBarrier();
	GroupMemoryBarrierWithGroupSync();
}
