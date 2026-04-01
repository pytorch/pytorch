// uav-write.slang
//TEST:SIMPLE:

// Just confirming that code that writes to a UAV will type-check.

RWTexture2D<float4> gOutput;

float4 test(uint2 coord, float4 value)
{
	// read
    value = value + gOutput[coord];

    // write
    gOutput[coord] = value;

    // read-modify-write
    gOutput[coord] += value;
	
	return value;
}