// function-static-const.slang

//TEST:CROSS_COMPILE:-target dxbc-assembly  -entry main -stage fragment -profile sm_5_0

// This test ensures that we compile `static const` variables inside
// of functions to reasonable HLSL output so that we don't introduce
// unexpected overhead by treating them as just `static`.
//

int test(int val)
{
	static const int kArray[] = {
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16
	};
	return kArray[val];
}

cbuffer C
{
    int index;
}

float4 main() : SV_Target
{
	return test(index);
}
